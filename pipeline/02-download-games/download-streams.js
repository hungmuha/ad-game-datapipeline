const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegStatic = require('ffmpeg-static');

// Set ffmpeg path
ffmpeg.setFfmpegPath(ffmpegStatic);

const MASTER_CSV = 'master.csv';
const VIDEOS_DIR = './videos';

// Statistics
const stats = {
  total: 0,
  skipped: 0,
  skippedDRM: 0,
  downloaded: 0,
  failed: 0
};

/**
 * Sanitize filename by removing special characters
 */
function sanitizeFilename(text) {
  return text
    .replace(/[^a-zA-Z0-9_-]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
}

/**
 * Read master CSV and return a Set of unique keys (title_airdate)
 */
async function readMasterCSV() {
  const masterEntries = new Set();

  if (!fs.existsSync(MASTER_CSV)) {
    console.log('Master CSV not found. Will create a new one.');
    return masterEntries;
  }

  return new Promise((resolve, reject) => {
    fs.createReadStream(MASTER_CSV)
      .pipe(csv())
      .on('data', (row) => {
        const key = `${row.title}_${row.air_date}`;
        masterEntries.add(key);
      })
      .on('end', () => {
        console.log(`Loaded ${masterEntries.size} entries from master CSV.`);
        resolve(masterEntries);
      })
      .on('error', reject);
  });
}

/**
 * Read input CSV and return array of rows
 */
async function readInputCSV(inputFile) {
  const rows = [];

  if (!fs.existsSync(inputFile)) {
    throw new Error(`Input file not found: ${inputFile}`);
  }

  return new Promise((resolve, reject) => {
    fs.createReadStream(inputFile)
      .pipe(csv())
      .on('data', (row) => {
        rows.push(row);
      })
      .on('end', () => {
        console.log(`Read ${rows.length} entries from input CSV.`);
        resolve(rows);
      })
      .on('error', reject);
  });
}

/**
 * Validate that required columns exist in the CSV data
 */
function validateCSVColumns(rows) {
  if (rows.length === 0) {
    throw new Error('Input CSV is empty.');
  }

  const firstRow = rows[0];
  const requiredColumns = ['title', 'air_date', 'stream_url'];
  const missingColumns = requiredColumns.filter(col => !(col in firstRow));

  if (missingColumns.length > 0) {
    throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
  }

  console.log('CSV validation passed.');
}

/**
 * Check if URL is MPD (DASH) format which typically has DRM
 */
function isMPDStream(streamUrl) {
  return streamUrl.toLowerCase().includes('.mpd');
}

/**
 * Download stream using ffmpeg
 */
function downloadStream(streamUrl, outputPath) {
  return new Promise((resolve, reject) => {
    console.log(`Downloading: ${outputPath}`);
    const startTime = Date.now();
    let lastProgressTime = startTime;
    let lastPercent = 0;
    let lastFileSize = 0;
    let lastSizeCheckTime = startTime;
    let progressInterval;

    // Extract referer from stream URL (for fubo.tv)
    let referer = 'https://www.fubo.tv/';
    try {
      const urlObj = new URL(streamUrl);
      // referer = `${urlObj.protocol}//${urlObj.host}/`;
    } catch (e) {
      console.warn('Could not parse URL for referer, using default');
    }

    // Build headers string for ffmpeg
    const headers = [
      `User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36`,
      `Referer: ${referer}`,
      `Accept: */*`,
      `Accept-Language: en-US,en;q=0.9`,
      `Origin: https://www.fubo.tv`
    ].join('\r\n');

    // Enhanced progress reporting with speed calculation
    const updateProgress = () => {
      const now = Date.now();
      const elapsed = Math.floor((now - startTime) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      
      let fileSize = 0;
      if (fs.existsSync(outputPath)) {
        try {
          fileSize = fs.statSync(outputPath).size;
        } catch (e) {
          // File might be locked, ignore
        }
      }
      
      const sizeMB = (fileSize / (1024 * 1024)).toFixed(2);
      
      // Calculate download speed
      let speedInfo = '';
      if (fileSize > lastFileSize && now > lastSizeCheckTime) {
        const bytesPerSecond = (fileSize - lastFileSize) / ((now - lastSizeCheckTime) / 1000);
        const mbps = (bytesPerSecond / (1024 * 1024)).toFixed(2);
        speedInfo = ` | Speed: ${mbps} MB/s`;
        lastFileSize = fileSize;
        lastSizeCheckTime = now;
      }
      
      // Build status line
      let statusLine = `\r[${minutes}m ${seconds}s] Size: ${sizeMB} MB${speedInfo}`;
      
      if (lastPercent > 0) {
        statusLine += ` | Progress: ${lastPercent.toFixed(2)}%`;
      } else {
        statusLine += ` | Downloading...`;
      }
      
      process.stdout.write(statusLine);
    };

    // Heartbeat to show activity and calculate speed
    progressInterval = setInterval(updateProgress, 2000); // Update every 2 seconds

    const ffmpegProcess = ffmpeg(streamUrl)
      .inputOptions([
        '-user_agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '-headers', headers,
        '-protocol_whitelist', 'file,http,https,tcp,tls,crypto', // Allow all protocols
        '-reconnect', '1', // Auto-reconnect on errors
        '-reconnect_at_eof', '1',
        '-reconnect_streamed', '1',
        '-reconnect_delay_max', '2' // Max 2 seconds between reconnects
      ])
      .outputOptions([
        '-c copy',
        '-bsf:a', 'aac_adtstoasc' // Fix audio stream issues
      ])
      .output(outputPath)
      .on('start', (commandLine) => {
        console.log(`FFmpeg command: ${commandLine}`);
        console.log('Starting download...');
      })
      .on('progress', (progress) => {
        lastProgressTime = Date.now();
        if (progress.percent) {
          lastPercent = progress.percent;
        }
        // Update progress display immediately when we get progress events
        updateProgress();
      })
      .on('end', () => {
        clearInterval(progressInterval);
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        let fileSize = 0;
        if (fs.existsSync(outputPath)) {
          try {
            fileSize = fs.statSync(outputPath).size;
          } catch (e) {
            // Ignore
          }
        }
        const sizeMB = (fileSize / (1024 * 1024)).toFixed(2);
        const avgSpeed = elapsed > 0 ? (fileSize / elapsed / (1024 * 1024)).toFixed(2) : '0';
        console.log(`\n✓ Download completed successfully in ${minutes}m ${seconds}s (${sizeMB} MB, avg ${avgSpeed} MB/s)`);
        resolve();
      })
      .on('error', (err) => {
        clearInterval(progressInterval);
        console.error(`\n✗ Download failed: ${err.message}`);
        reject(err);
      });

    // Set a timeout to detect if download is stuck (2 hours)
    const timeout = setTimeout(() => {
      if (Date.now() - lastProgressTime > 300000) { // 5 minutes without progress
        console.error('\n✗ Download appears to be stuck (no progress for 5 minutes)');
        ffmpegProcess.kill();
        reject(new Error('Download timeout - no progress for 5 minutes'));
      }
    }, 7200000); // 2 hour overall timeout

    ffmpegProcess.on('end', () => clearTimeout(timeout));
    ffmpegProcess.on('error', () => clearTimeout(timeout));

    ffmpegProcess.run();
  });
}

/**
 * Append entry to master CSV
 */
async function appendToMasterCSV(entry) {
  const fileExists = fs.existsSync(MASTER_CSV);

  const csvWriter = createObjectCsvWriter({
    path: MASTER_CSV,
    header: [
      { id: 'title', title: 'title' },
      { id: 'air_date', title: 'air_date' },
      { id: 'network', title: 'network' },
      { id: 'stream_url', title: 'stream_url' },
      { id: 'status', title: 'status' },
      { id: 'file_path', title: 'file_path' },
      { id: 'filename', title: 'filename' },
      { id: 'extraction_time', title: 'extraction_time' },
      { id: 'download_time', title: 'download_time' }
    ],
    append: fileExists
  });

  await csvWriter.writeRecords([entry]);
}

/**
 * Process a single entry
 */
async function processEntry(row) {
  const { title, air_date, stream_url, network, extraction_time, status: originalStatus } = row;
  const sanitizedTitle = sanitizeFilename(title);
  const sanitizedDate = sanitizeFilename(air_date);
  const filename = `${sanitizedTitle}_${sanitizedDate}.mp4`;
  const outputPath = path.join(VIDEOS_DIR, filename);
  const absolutePath = path.resolve(outputPath);

  const entry = {
    title,
    air_date,
    network: network || '',
    stream_url,
    status: 'pending',
    file_path: '',
    filename: '',
    extraction_time: extraction_time || '',
    download_time: new Date().toISOString()
  };

  // Check if MPD (DRM-protected) stream
  if (isMPDStream(stream_url)) {
    console.log('Skipping: MPD stream detected (DRM-protected)');
    entry.status = 'skipped_drm';
    stats.skippedDRM++;
    await appendToMasterCSV(entry);
    return;
  }

  try {
    await downloadStream(stream_url, outputPath);
    entry.status = 'downloaded';
    entry.file_path = absolutePath;
    entry.filename = filename;
    stats.downloaded++;
  } catch (error) {
    entry.status = 'failed';
    stats.failed++;
  }

  await appendToMasterCSV(entry);
}

/**
 * Prevent system sleep using caffeinate (macOS)
 */
function preventSleep() {
  if (process.platform === 'darwin') {
    console.log('Preventing system sleep during download...');
    const caffeinate = spawn('caffeinate', ['-d', '-i', '-m', '-s']);
    
    // Cleanup on exit
    process.on('exit', () => {
      caffeinate.kill();
    });
    process.on('SIGINT', () => {
      caffeinate.kill();
      process.exit(0);
    });
    process.on('SIGTERM', () => {
      caffeinate.kill();
      process.exit(0);
    });
    
    return caffeinate;
  }
  return null;
}

/**
 * Main function
 */
async function main() {
  try {
    // Parse command line arguments
    const args = process.argv.slice(2);
    if (args.length === 0) {
      console.error('Usage: node download-streams.js <input.csv>');
      process.exit(1);
    }

    const inputFile = args[0];
    console.log(`Input file: ${inputFile}\n`);

    // Prevent system sleep
    const caffeinateProcess = preventSleep();

    // Create videos directory if it doesn't exist
    if (!fs.existsSync(VIDEOS_DIR)) {
      fs.mkdirSync(VIDEOS_DIR);
      console.log(`Created directory: ${VIDEOS_DIR}`);
    }

    // Read master CSV
    const masterEntries = await readMasterCSV();

    // Read and validate input CSV
    const inputRows = await readInputCSV(inputFile);
    validateCSVColumns(inputRows);

    // Filter entries that haven't been processed
    const toProcess = [];
    for (const row of inputRows) {
      stats.total++;
      const key = `${row.title}_${row.air_date}`;

      if (masterEntries.has(key)) {
        console.log(`Skipping (already processed): ${row.title} - ${row.air_date}`);
        stats.skipped++;
      } else {
        toProcess.push(row);
      }
    }

    console.log(`\nFound ${toProcess.length} new entries to process.\n`);

    // Process each entry
    for (let i = 0; i < toProcess.length; i++) {
      console.log(`\n[${i + 1}/${toProcess.length}] Processing: ${toProcess[i].title} - ${toProcess[i].air_date}`);
      await processEntry(toProcess[i]);
    }

    // Display summary
    console.log('\n' + '='.repeat(50));
    console.log('SUMMARY');
    console.log('='.repeat(50));
    console.log(`Total entries:           ${stats.total}`);
    console.log(`Skipped (existing):      ${stats.skipped}`);
    console.log(`Skipped (DRM-protected): ${stats.skippedDRM}`);
    console.log(`Successfully downloaded: ${stats.downloaded}`);
    console.log(`Failed downloads:        ${stats.failed}`);
    console.log('='.repeat(50));

    // Cleanup caffeinate
    if (caffeinateProcess) {
      caffeinateProcess.kill();
      console.log('\nSystem sleep prevention disabled.');
    }

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Run main function
main();
