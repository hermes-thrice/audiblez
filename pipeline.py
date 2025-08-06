#!/usr/bin/env python3
"""
Audiobook Pipeline Management System
Handles batch processing of audiobooks from Supabase storage
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configuration
AUDIOBOOK_GENERATION_COMMAND = "audiblez {epub_path} -v af_heart --cuda"

class AudiobookPipeline:
    def __init__(self):
        """Initialize the pipeline with Supabase client and configuration"""
        self.setup_logging()
        self.setup_supabase()
        self.temp_dir = Path("./epubs")
        self.temp_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_supabase(self):
        """Initialize Supabase client with credentials from .env"""
        try:
            supabase_url = os.getenv('NEXT_PUBLIC_SUPABASE_URL')
            supabase_anon_key = os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
            service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            
            if not all([supabase_url, supabase_anon_key, service_role_key]):
                raise ValueError("Missing Supabase credentials in .env file")
                
            # Use service role key for admin operations
            self.supabase: Client = create_client(supabase_url, service_role_key)
            self.logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Supabase client: {e}")
            sys.exit(1)
            
    def fetch_all_books(self) -> List[Dict]:
        """Fetch all books from the 'books' table"""
        try:
            response = self.supabase.table('books').select('*').execute()
            books = response.data
            self.logger.info(f"Fetched {len(books)} books from database")
            return books
        except Exception as e:
            self.logger.error(f"Failed to fetch books: {e}")
            return []
            
    def download_epub_from_bucket(self, slug: str) -> Optional[Path]:
        """Download epub file from 'downloads' bucket by slug"""
        try:
            # Download file from downloads/slug/slug.epub
            file_path = f"{slug}/{slug}.epub"
            local_path = self.temp_dir / f"{slug}.epub"
            
            # Download file from Supabase storage
            response = self.supabase.storage.from_("downloads").download(file_path)
            
            # Write to local file
            with open(local_path, 'wb') as f:
                f.write(response)
                
            self.logger.info(f"Downloaded {file_path} to {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download epub for slug '{slug}': {e}")
            return None
            
    def upload_m4b_to_bucket(self, slug: str, m4b_path: Path) -> Optional[str]:
        """Upload m4b file to 'audiobooks' bucket and return public URL"""
        try:
            # Upload to audiobooks/slug/slug.m4b
            bucket_path = f"{slug}/{slug}.m4b"
            
            with open(m4b_path, 'rb') as f:
                file_data = f.read()
                
            # Upload file (this will overwrite if exists)
            self.supabase.storage.from_("audiobooks").upload(
                bucket_path, file_data, {"upsert": "true"}
            )
            
            # Get public URL
            public_url = self.supabase.storage.from_("audiobooks").get_public_url(bucket_path)
            
            self.logger.info(f"Uploaded {m4b_path} to {bucket_path}")
            return public_url
            
        except Exception as e:
            self.logger.error(f"Failed to upload m4b for slug '{slug}': {e}")
            return None
    
    def list_audiobook_folders(self) -> List[str]:
        """List all folders in the audiobooks bucket"""
        try:
            response = self.supabase.storage.from_("audiobooks").list()
            
            # Handle case where response might be None or have different structure
            if not response:
                self.logger.warning("Empty response from storage list")
                return []
            
            # Filter for folders (items without file extensions and where metadata is None or has no mimetype)
            folders = []
            for item in response:
                if item and isinstance(item, dict):
                    name = item.get('name', '')
                    metadata = item.get('metadata')
                    
                    # Folders typically don't have file extensions and have metadata=None
                    # or metadata without mimetype
                    has_mimetype = False
                    if metadata and isinstance(metadata, dict):
                        has_mimetype = bool(metadata.get('mimetype'))
                    
                    if name and '.' not in name and not has_mimetype:
                        folders.append(name)
            
            self.logger.info(f"Found {len(folders)} folders in audiobooks bucket: {folders}")
            return folders
        except Exception as e:
            self.logger.error(f"Failed to list audiobook folders: {e}")
            return []
    
    def check_mp3_exists(self, slug: str) -> bool:
        """Check if MP3 file exists in the audiobooks bucket folder"""
        try:
            file_path = f"{slug}/{slug}.mp3"
            # Try to get file info - if it exists, this won't raise an exception
            self.supabase.storage.from_("audiobooks").get_public_url(file_path)
            # Additional check by trying to list the specific file
            response = self.supabase.storage.from_("audiobooks").list(slug)
            mp3_files = [item for item in response if item['name'] == f"{slug}.mp3"]
            return len(mp3_files) > 0
        except Exception:
            return False
    
    def check_m4b_exists(self, slug: str) -> bool:
        """Check if M4B file exists in the audiobooks bucket folder"""
        try:
            response = self.supabase.storage.from_("audiobooks").list(slug)
            m4b_files = [item for item in response if item['name'] == f"{slug}.m4b"]
            return len(m4b_files) > 0
        except Exception:
            return False
    
    def download_m4b_from_bucket(self, slug: str) -> Optional[Path]:
        """Download M4B file from audiobooks bucket"""
        try:
            file_path = f"{slug}/{slug}.m4b"
            local_path = self.temp_dir / f"{slug}.m4b"
            
            response = self.supabase.storage.from_("audiobooks").download(file_path)
            
            with open(local_path, 'wb') as f:
                f.write(response)
                
            self.logger.info(f"Downloaded {file_path} to {local_path}")
            return local_path
            
        except Exception as e:
            self.logger.error(f"Failed to download M4B for slug '{slug}': {e}")
            return None
    
    def convert_m4b_to_mp3(self, m4b_path: Path) -> Optional[Path]:
        """Convert M4B file to MP3 using ffmpeg"""
        try:
            mp3_path = m4b_path.parent / f"{m4b_path.stem}.mp3"
            
            # Use ffmpeg with libmp3lame for optimal speech compression (target: 10-12MB/hour)
            command = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(m4b_path),  # Input M4B file
                '-c:a', 'libmp3lame',  # Use LAME MP3 encoder for better compression
                '-b:a', '48k',  # Aggressive bitrate for speech (was 64k)
                '-ar', '22050',  # Optimal sample rate for speech
                '-ac', '1',  # Mono audio (50% size reduction)
                '-q:a', '4',  # VBR quality (0=best, 9=worst, 4=good balance)
                '-compression_level', '9',  # Maximum compression effort
                '-reservoir', 'true',  # Enable bit reservoir for better quality
                '-joint_stereo', 'false',  # Disable since we're mono anyway
                '-movflags', '+faststart',  # Optimize for streaming
                '-f', 'mp3',  # Output format
                str(mp3_path)  # Output file
            ]
            
            self.logger.info(f"Converting M4B to MP3: {m4b_path} -> {mp3_path}")
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0 and mp3_path.exists():
                # Validate file size and calculate compression ratio
                file_size_mb = mp3_path.stat().st_size / (1024 * 1024)
                
                # Estimate duration from M4B file for size validation
                try:
                    duration_result = subprocess.run([
                        'ffprobe', '-i', str(m4b_path), '-show_entries', 'format=duration', 
                        '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1'
                    ], capture_output=True, text=True, check=True)
                    duration_hours = float(duration_result.stdout.strip()) / 3600
                    mb_per_hour = file_size_mb / duration_hours if duration_hours > 0 else 0
                    
                    self.logger.info(f"Successfully converted to MP3: {mp3_path}")
                    self.logger.info(f"MP3 file size: {file_size_mb:.1f}MB, Duration: {duration_hours:.2f}h, Ratio: {mb_per_hour:.1f}MB/hour")
                    
                    # Warn if file size exceeds target (12MB/hour)
                    if mb_per_hour > 12:
                        self.logger.warning(f"MP3 file size ({mb_per_hour:.1f}MB/hour) exceeds target of 10-12MB/hour")
                    
                except Exception as e:
                    self.logger.warning(f"Could not validate MP3 file size: {e}")
                    self.logger.info(f"Successfully converted to MP3: {mp3_path} ({file_size_mb:.1f}MB)")
                
                return mp3_path
            else:
                self.logger.error(f"MP3 conversion failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"MP3 conversion timed out for {m4b_path}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to convert M4B to MP3: {e}")
            return None
    
    def upload_mp3_to_bucket(self, slug: str, mp3_path: Path) -> Optional[str]:
        """Upload MP3 file to audiobooks bucket and return public URL"""
        try:
            bucket_path = f"{slug}/{slug}.mp3"
            
            # Log file size for debugging
            file_size = mp3_path.stat().st_size
            self.logger.info(f"Preparing to upload MP3: {mp3_path} (size: {file_size / (1024*1024):.1f} MB)")
            
            self.logger.info(f"Reading MP3 file data...")
            with open(mp3_path, 'rb') as f:
                file_data = f.read()
            
            self.logger.info(f"File data read successfully, starting upload to {bucket_path}...")
            
            # Upload file (this will overwrite if exists)
            response = self.supabase.storage.from_("audiobooks").upload(
                bucket_path, file_data, {"upsert": "true"}
            )
            
            self.logger.info(f"Upload response received: {response}")
            
            # Get public URL
            self.logger.info(f"Getting public URL for {bucket_path}...")
            public_url = self.supabase.storage.from_("audiobooks").get_public_url(bucket_path)
            
            self.logger.info(f"Successfully uploaded {mp3_path} to {bucket_path} (public URL: {public_url})")
            return public_url
            
        except Exception as e:
            self.logger.error(f"Failed to upload MP3 for slug '{slug}': {e}")
            import traceback
            self.logger.error(f"Upload error traceback: {traceback.format_exc()}")
            return None
            
    def update_audiobook_url(self, slug: str, audiobook_url: str) -> bool:
        """Update the audiobook_url column in the books table"""
        try:
            response = self.supabase.table('books').update({
                'audiobook_url': audiobook_url
            }).eq('slug', slug).execute()
            
            if response.data:
                self.logger.info(f"Updated audiobook_url for slug '{slug}'")
                return True
            else:
                self.logger.warning(f"No book found with slug '{slug}' to update")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update audiobook_url for slug '{slug}': {e}")
            return False
    
    def update_audiobook_mp3_url(self, slug: str, mp3_url: str) -> bool:
        """Update the audiobook_mp3_url column in the books table"""
        try:
            response = self.supabase.table('books').update({
                'audiobook_mp3_url': mp3_url
            }).eq('slug', slug).execute()
            
            if response.data:
                self.logger.info(f"Updated audiobook_mp3_url for slug '{slug}'")
                return True
            else:
                self.logger.warning(f"No book found with slug '{slug}' to update MP3 URL")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to update audiobook_mp3_url for slug '{slug}': {e}")
            return False
    
    def get_book_id_by_slug(self, slug: str) -> Optional[str]:
        """Get book UUID from books table by slug"""
        try:
            response = self.supabase.table('books').select('id').eq('slug', slug).execute()
            if response.data and len(response.data) > 0:
                book_id = response.data[0]['id']
                self.logger.info(f"Found book_id {book_id} for slug '{slug}'")
                return book_id
            else:
                self.logger.warning(f"No book found with slug '{slug}'")
                return None
        except Exception as e:
            self.logger.error(f"Failed to get book_id for slug '{slug}': {e}")
            return None
    
    def clear_existing_chapters(self, book_id: str) -> bool:
        """Clear existing chapter data for a book (for regeneration)"""
        try:
            self.supabase.table('audiobook_chapters').delete().eq('book_id', book_id).execute()
            self.logger.info(f"Cleared existing chapters for book_id {book_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear existing chapters for book_id {book_id}: {e}")
            return False
    
    def extract_chapter_timing_from_final_audio(self, m4b_path: Path, chapter_titles: List[str]) -> List[Dict]:
        """Extract accurate chapter timing from final M4B file to prevent drift"""
        try:
            import subprocess
            import json
            
            self.logger.info(f"Analyzing final M4B file for accurate chapter timing: {m4b_path}")
            
            # Step 1: Extract chapter metadata from M4B file
            chapter_info_cmd = [
                'ffprobe', '-i', str(m4b_path), 
                '-print_format', 'json', '-show_chapters', 
                '-v', 'quiet'
            ]
            
            result = subprocess.run(chapter_info_cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)
            
            chapter_data = []
            chapters = metadata.get('chapters', [])
            
            # Step 2: Use embedded chapter markers if available
            if chapters:
                self.logger.info(f"Found {len(chapters)} embedded chapters in M4B file")
                self.logger.info(f"DEBUG: Raw chapter metadata from M4B:")
                for i, chapter_meta in enumerate(chapters):
                    self.logger.info(f"DEBUG: Chapter {i}: {chapter_meta}")
                    
                    original_start = float(chapter_meta.get('start_time', 0))
                    original_end = float(chapter_meta.get('end_time', 0))
                    start_time_sec = int(original_start)
                    end_time_sec = int(original_end)
                    
                    self.logger.info(f"DEBUG: Chapter {i+1} - Original: {original_start:.3f}s -> {original_end:.3f}s")
                    self.logger.info(f"DEBUG: Chapter {i+1} - Converted: {start_time_sec}s -> {end_time_sec}s")
                    
                    # Apply 2-second buffer for chapters after the first
                    buffered_start = start_time_sec
                    if i > 0:
                        buffered_start = max(0, start_time_sec - 2)
                        self.logger.info(f"DEBUG: Chapter {i+1} - Applied 2s buffer: {start_time_sec}s -> {buffered_start}s")
                    
                    chapter_title = chapter_titles[i] if i < len(chapter_titles) else f"Chapter {i + 1}"
                    
                    chapter_info = {
                        'chapter_number': i + 1,
                        'title': chapter_title,
                        'start_time_seconds': buffered_start,
                        'end_time_seconds': end_time_sec
                    }
                    chapter_data.append(chapter_info)
                    self.logger.info(f"DEBUG: Final chapter {i+1}: '{chapter_title}' -> {buffered_start}s - {end_time_sec}s")
                    
            else:
                # Step 3: Fallback - detect chapter boundaries using silence detection
                self.logger.info("No embedded chapters found, using silence detection for chapter boundaries")
                chapter_data = self.detect_chapter_boundaries_with_silence(m4b_path, chapter_titles)
            
            self.logger.info(f"Extracted accurate timing data for {len(chapter_data)} chapters from final M4B")
            return chapter_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract timing from final M4B: {e}")
            # Fallback to WAV-based timing if M4B analysis fails
            self.logger.info("Falling back to WAV-based timing estimation")
            return self.extract_chapter_timing_from_wavs_with_buffer(chapter_titles)
    
    def detect_chapter_boundaries_with_silence(self, m4b_path: Path, chapter_titles: List[str]) -> List[Dict]:
        """Detect chapter boundaries using silence detection on final audio"""
        try:
            import subprocess
            
            # Detect silence periods (likely chapter boundaries)
            silence_cmd = [
                'ffmpeg', '-i', str(m4b_path), '-af', 
                'silencedetect=n=-30dB:d=1.0', '-f', 'null', '-'
            ]
            
            result = subprocess.run(silence_cmd, capture_output=True, text=True)
            silence_output = result.stderr
            
            # Parse silence detection output
            silence_times = []
            for line in silence_output.split('\n'):
                if 'silence_end' in line:
                    # Extract timestamp: "silence_end: 123.456"
                    parts = line.split('silence_end: ')
                    if len(parts) > 1:
                        try:
                            timestamp = float(parts[1].split(' ')[0])
                            silence_times.append(int(timestamp))
                        except (ValueError, IndexError):
                            continue
            
            # Build chapter data with detected boundaries
            chapter_data = []
            start_time_sec = 0
            
            for i, title in enumerate(chapter_titles):
                # Use next silence boundary or end of file
                if i < len(silence_times):
                    end_time_sec = silence_times[i]
                else:
                    # Get total duration for last chapter
                    duration_cmd = ['ffprobe', '-i', str(m4b_path), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
                    duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                    end_time_sec = int(float(duration_result.stdout.strip()))
                
                # Apply buffer for chapters after the first
                buffered_start = max(0, start_time_sec - 2) if i > 0 else start_time_sec
                
                chapter_info = {
                    'chapter_number': i + 1,
                    'title': title,
                    'start_time_seconds': buffered_start,
                    'end_time_seconds': end_time_sec
                }
                chapter_data.append(chapter_info)
                start_time_sec = end_time_sec
                
            self.logger.info(f"Detected {len(chapter_data)} chapter boundaries using silence analysis")
            return chapter_data
            
        except Exception as e:
            self.logger.error(f"Silence detection failed: {e}")
            return self.extract_chapter_timing_from_wavs_with_buffer(chapter_titles)
    
    def extract_chapter_timing_from_wavs_with_buffer(self, chapter_titles: List[str]) -> List[Dict]:
        """Fallback: Extract timing from WAV files with drift compensation buffer"""
        try:
            # This is the old method but with buffer compensation
            self.logger.warning("Using WAV-based timing estimation with buffer compensation")
            
            chapter_data = []
            start_time_sec = 0
            
            for i, title in enumerate(chapter_titles):
                # Estimate 3-minute average chapter length if we can't probe
                estimated_duration = 180  # 3 minutes default
                end_time_sec = start_time_sec + estimated_duration
                
                # Apply buffer for chapters after the first
                buffered_start = max(0, start_time_sec - 2) if i > 0 else start_time_sec
                
                chapter_info = {
                    'chapter_number': i + 1,
                    'title': title,
                    'start_time_seconds': buffered_start,
                    'end_time_seconds': end_time_sec
                }
                chapter_data.append(chapter_info)
                start_time_sec = end_time_sec
            
            return chapter_data
            
        except Exception as e:
            self.logger.error(f"WAV timing fallback failed: {e}")
            return []
    
    def insert_chapter_data(self, book_id: str, chapter_data: List[Dict]) -> bool:
        """Insert chapter data into audiobook_chapters table"""
        try:
            # Prepare records for batch insert
            records = []
            for chapter in chapter_data:
                record = {
                    'book_id': book_id,
                    'chapter_number': chapter['chapter_number'],
                    'title': chapter['title'],
                    'start_time_seconds': chapter['start_time_seconds'],
                    'end_time_seconds': chapter['end_time_seconds']
                }
                records.append(record)
            
            # Batch insert all chapters
            response = self.supabase.table('audiobook_chapters').insert(records).execute()
            
            if response.data:
                self.logger.info(f"Successfully inserted {len(records)} chapters for book_id {book_id}")
                return True
            else:
                self.logger.error(f"Failed to insert chapter data - no data returned")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to insert chapter data for book_id {book_id}: {e}")
            import traceback
            self.logger.error(f"Chapter insert error traceback: {traceback.format_exc()}")
            return False
    
    def generate_and_upload_mp3(self, slug: str, m4b_path: Path) -> bool:
        """Generate MP3 from M4B and upload to bucket, update database"""
        mp3_path = None
        try:
            # Step 1: Convert M4B to MP3
            self.logger.info(f"Converting M4B to MP3 for {slug}")
            mp3_path = self.convert_m4b_to_mp3(m4b_path)
            if not mp3_path:
                self.logger.error(f"Failed to convert M4B to MP3 for {slug}")
                return False
            
            # Step 2: Upload MP3 to bucket
            self.logger.info(f"Uploading MP3 for {slug}")
            mp3_url = self.upload_mp3_to_bucket(slug, mp3_path)
            if not mp3_url:
                self.logger.error(f"Failed to upload MP3 for {slug}")
                return False
            
            # Step 3: Update database
            self.logger.info(f"Updating database MP3 URL for {slug}")
            if not self.update_audiobook_mp3_url(slug, mp3_url):
                self.logger.error(f"Failed to update database MP3 URL for {slug}")
                return False
            
            self.logger.info(f"Successfully generated and uploaded MP3 for {slug}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating MP3 for {slug}: {e}")
            return False
            
        finally:
            # Clean up temporary MP3 file
            if mp3_path and mp3_path.exists():
                mp3_path.unlink()
                self.logger.info(f"Cleaned up {mp3_path}")
            # Note: M4B cleanup is handled by the calling function
            
    def generate_audiobook(self, epub_path: Path) -> tuple[Optional[Path], List[Dict]]:
        """Run audiobook generation command and return path to generated m4b and chapter data"""
        try:
            # Use absolute path for the command
            abs_epub_path = epub_path.resolve()
            command = AUDIOBOOK_GENERATION_COMMAND.format(epub_path=abs_epub_path)
            
            self.logger.info(f"Running audiobook generation: {command}")
            
            # Run the command from the epub directory
            result = subprocess.run(
                command.split(),
                cwd=abs_epub_path.parent,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                # Look for generated m4b file
                m4b_path = epub_path.parent / f"{epub_path.stem}.m4b"
                if m4b_path.exists():
                    self.logger.info(f"Audiobook generated successfully: {m4b_path}")
                    
                    # Extract chapter titles from generated WAV files
                    chapter_titles = self.extract_chapter_data_from_output_dir(abs_epub_path.parent, epub_path.stem)
                    
                    return m4b_path, chapter_titles
                else:
                    self.logger.error(f"Command succeeded but m4b file not found: {m4b_path}")
                    return None, []
            else:
                self.logger.error(f"Audiobook generation failed: {result.stderr}")
                return None, []
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Audiobook generation timed out for {epub_path}")
            return None, []
        except Exception as e:
            self.logger.error(f"Failed to generate audiobook for {epub_path}: {e}")
            return None, []
    
    def extract_chapter_data_from_output_dir(self, output_dir: Path, epub_stem: str) -> List[str]:
        """Extract chapter data from the output directory after audiobook generation"""
        try:
            # Find all WAV files generated by audiblez
            wav_pattern = f"{epub_stem}_chapter_*_*.wav"
            wav_files = list(output_dir.glob(wav_pattern))
            
            self.logger.info(f"DEBUG: Looking for WAV files with pattern: {wav_pattern}")
            self.logger.info(f"DEBUG: Found {len(wav_files)} WAV files:")
            for wav_file in wav_files:
                self.logger.info(f"DEBUG: - {wav_file.name}")
            
            # Sort by chapter number (extract from filename)
            def get_chapter_num(wav_file):
                # File format: {epub_stem}_chapter_{num}_{voice}_{chapter_name}.wav
                parts = wav_file.stem.split('_chapter_')
                if len(parts) > 1:
                    chapter_part = parts[1].split('_')[0]
                    try:
                        return int(chapter_part)
                    except ValueError:
                        return 999  # Put non-numeric at end
                return 999
            
            wav_files.sort(key=get_chapter_num)
            
            self.logger.info(f"DEBUG: WAV files after sorting:")
            for i, wav_file in enumerate(wav_files):
                self.logger.info(f"DEBUG: [{i}] {wav_file.name} -> chapter #{get_chapter_num(wav_file)}")
            
            if not wav_files:
                self.logger.warning(f"No WAV files found in {output_dir} with pattern {wav_pattern}")
                return []
            
            # Extract chapter titles from filenames 
            # Format: {epub_stem}_chapter_{num}_{voice}_{chapter_name}.wav
            chapter_titles = []
            for wav_file in wav_files:
                chapter_num = get_chapter_num(wav_file)
                
                self.logger.info(f"DEBUG: Processing {wav_file.name}")
                self.logger.info(f"DEBUG: Chapter number: {chapter_num}")
                
                # Try to extract title from filename
                title = None
                parts = wav_file.stem.split('_chapter_')
                self.logger.info(f"DEBUG: Split on '_chapter_': {parts}")
                
                if len(parts) > 1:
                    # Split the part after '_chapter_': {num}_{voice}_{chapter_name}
                    remaining = parts[1]
                    components = remaining.split('_')
                    self.logger.info(f"DEBUG: Components after chapter: {components}")
                    
                    # Should have at least: [num, voice, ...title_parts]
                    if len(components) >= 3:
                        # Skip chapter number (index 0) and voice (index 1), take rest as title
                        title_parts = components[2:]
                        title = '_'.join(title_parts)
                        self.logger.info(f"DEBUG: Raw title parts: {title_parts}")
                        self.logger.info(f"DEBUG: Joined raw title: '{title}'")
                        
                        # Clean up the title
                        title = title.replace('_', ' ').strip()
                        self.logger.info(f"DEBUG: After underscore replacement: '{title}'")
                        
                        # Remove file extensions
                        if title.lower().endswith('.xhtml'):
                            title = title[:-6]
                            self.logger.info(f"DEBUG: After removing .xhtml: '{title}'")
                        if title.lower().endswith('.html'):
                            title = title[:-5]
                            self.logger.info(f"DEBUG: After removing .html: '{title}'")
                        
                        # Convert to title case, but preserve important formatting
                        if title and not title.isupper():
                            original_title = title
                            title = title.title()
                            self.logger.info(f"DEBUG: Title case conversion: '{original_title}' -> '{title}'")
                        
                        # Remove voice artifacts that might have slipped through
                        voice_patterns = ['af_heart', 'af_sky', 'am_heart', 'am_sky']
                        for pattern in voice_patterns:
                            if title.lower().startswith(pattern):
                                old_title = title
                                title = title[len(pattern):].strip()
                                self.logger.info(f"DEBUG: Removed voice prefix '{pattern}': '{old_title}' -> '{title}'")
                            if title.lower().endswith(pattern):
                                old_title = title
                                title = title[:-len(pattern)].strip()
                                self.logger.info(f"DEBUG: Removed voice suffix '{pattern}': '{old_title}' -> '{title}'")
                    else:
                        self.logger.warning(f"DEBUG: Not enough components ({len(components)}) to extract title from: {components}")
                else:
                    self.logger.warning(f"DEBUG: Could not split filename on '_chapter_'")
                
                # Use cleaned title or fallback
                if title and title.strip():
                    final_title = title.strip()
                    chapter_titles.append(final_title)
                    self.logger.info(f"DEBUG: Using extracted title: '{final_title}'")
                else:
                    # Fallback for chapter 0 (title) or when parsing fails
                    if chapter_num == 0:
                        fallback_title = "Introduction"
                    else:
                        fallback_title = f"Chapter {chapter_num}"
                    chapter_titles.append(fallback_title)
                    self.logger.warning(f"DEBUG: Using fallback title: '{fallback_title}' (extraction failed)")
                        
                self.logger.info(f"DEBUG: Final title for chapter {chapter_num}: '{chapter_titles[-1]}'")
                self.logger.info(f"DEBUG: ---")
            
            # Return chapter titles for now - timing will be extracted from M4B file
            self.logger.info(f"Extracted {len(chapter_titles)} chapter titles from output directory")
            return chapter_titles  # Return titles, not chapter_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract chapter data from output dir: {e}")
            return []
            
    def cleanup_temp_files(self, epub_path: Path, m4b_path: Path = None):
        """Clean up temporary files"""
        try:
            if epub_path and epub_path.exists():
                epub_path.unlink()
                self.logger.info(f"Cleaned up {epub_path}")
                
            if m4b_path and m4b_path.exists():
                m4b_path.unlink()
                self.logger.info(f"Cleaned up {m4b_path}")
                
            # Also clean up any .wav files
            for wav_file in epub_path.parent.glob("*.wav"):
                wav_file.unlink()
                self.logger.info(f"Cleaned up {wav_file}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
            
    def process_single_book(self, slug: str) -> bool:
        """Process a single book by slug - core logic used by all commands"""
        self.logger.info(f"Starting processing for book: {slug}")
        
        epub_path = None
        m4b_path = None
        
        try:
            # Step 1: Download epub from downloads bucket
            self.logger.info(f"Downloading epub for {slug}")
            epub_path = self.download_epub_from_bucket(slug)
            if not epub_path:
                self.logger.error(f"Failed to download epub for {slug}")
                return False
                
            # Step 2: Generate audiobook
            self.logger.info(f"Generating audiobook for {slug}")
            m4b_path, chapter_titles = self.generate_audiobook(epub_path)
            if not m4b_path:
                self.logger.error(f"Failed to generate audiobook for {slug}")
                return False
                
            # Step 3: Upload m4b to audiobooks bucket
            self.logger.info(f"Uploading audiobook for {slug}")
            audiobook_url = self.upload_m4b_to_bucket(slug, m4b_path)
            if not audiobook_url:
                self.logger.error(f"Failed to upload audiobook for {slug}")
                return False
                
            # Step 4: Update database with audiobook URL
            self.logger.info(f"Updating database for {slug}")
            if not self.update_audiobook_url(slug, audiobook_url):
                self.logger.error(f"Failed to update database for {slug}")
                return False
            
            # Step 5: Generate and upload MP3
            self.logger.info(f"Generating MP3 for {slug}")
            if not self.generate_and_upload_mp3(slug, m4b_path):
                self.logger.warning(f"Failed to generate MP3 for {slug}, but M4B was successful")
                # Don't return False here - M4B generation was successful
            
            # Step 6: Extract accurate chapter timing and insert into database
            if chapter_titles:
                self.logger.info(f"Extracting accurate chapter timing for {slug}")
                # Use the new accurate timing extraction from final M4B
                chapter_data = self.extract_chapter_timing_from_final_audio(m4b_path, chapter_titles)
                
                if chapter_data:
                    book_id = self.get_book_id_by_slug(slug)
                    if book_id:
                        # Clear existing chapters for regeneration
                        self.clear_existing_chapters(book_id)
                        # Insert new chapter data with accurate timing
                        if not self.insert_chapter_data(book_id, chapter_data):
                            self.logger.warning(f"Failed to insert chapter data for {slug}, but audiobook generation was successful")
                    else:
                        self.logger.warning(f"Could not get book_id for {slug}, skipping chapter data insertion")
                else:
                    self.logger.warning(f"Failed to extract accurate timing for {slug}")
            else:
                self.logger.warning(f"No chapter titles extracted for {slug}")
                
            self.logger.info(f"Successfully processed book: {slug}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing book {slug}: {e}")
            return False
            
        finally:
            # Clean up temporary files (but keep M4B until after MP3 generation)
            if epub_path and epub_path.exists():
                epub_path.unlink()
                self.logger.info(f"Cleaned up {epub_path}")
            
            # Clean up any remaining wav files
            if epub_path:
                for wav_file in epub_path.parent.glob("*.wav"):
                    wav_file.unlink()
                    self.logger.info(f"Cleaned up {wav_file}")
            
            # M4B cleanup is handled in generate_and_upload_mp3 after MP3 generation
            if m4b_path and m4b_path.exists():
                m4b_path.unlink()
                self.logger.info(f"Cleaned up {m4b_path}")
                
    def cmd_regenerate_all(self):
        """Command: regenerate-all - Process all books regardless of current status"""
        self.logger.info("Starting regenerate-all command")
        
        books = self.fetch_all_books()
        if not books:
            self.logger.warning("No books found in database")
            return
            
        total_books = len(books)
        successful = 0
        failed = 0
        
        self.logger.info(f"Processing {total_books} books...")
        
        for i, book in enumerate(books, 1):
            slug = book.get('slug')
            if not slug:
                self.logger.warning(f"Book {i} has no slug, skipping")
                failed += 1
                continue
                
            self.logger.info(f"Processing book {i}/{total_books}: {slug}")
            
            if self.process_single_book(slug):
                successful += 1
                self.logger.info(f"✅ {slug} - Success ({successful}/{total_books})")
            else:
                failed += 1
                self.logger.error(f"❌ {slug} - Failed ({failed}/{total_books})")
                
        self.logger.info(f"Regenerate-all completed: {successful} successful, {failed} failed")
        
    def cmd_generate_new(self):
        """Command: generate-new - Process only books without audiobook_url"""
        self.logger.info("Starting generate-new command")
        
        books = self.fetch_all_books()
        if not books:
            self.logger.warning("No books found in database")
            return
            
        # Filter books that need processing (audiobook_url is NULL)
        books_to_process = [book for book in books if not book.get('audiobook_url')]
        
        if not books_to_process:
            self.logger.info("All books already have audiobook URLs - nothing to generate")
            return
            
        total_books = len(books_to_process)
        successful = 0
        failed = 0
        
        self.logger.info(f"Found {total_books} books without audiobooks to process...")
        
        for i, book in enumerate(books_to_process, 1):
            slug = book.get('slug')
            if not slug:
                self.logger.warning(f"Book {i} has no slug, skipping")
                failed += 1
                continue
                
            self.logger.info(f"Processing book {i}/{total_books}: {slug}")
            
            if self.process_single_book(slug):
                successful += 1
                self.logger.info(f"✅ {slug} - Success ({successful}/{total_books})")
            else:
                failed += 1
                self.logger.error(f"❌ {slug} - Failed ({failed}/{total_books})")
                
        self.logger.info(f"Generate-new completed: {successful} successful, {failed} failed")
    
    def cmd_generate_mp3_new(self):
        """Command: generate-mp3-new - Convert M4B files to MP3 where MP3 doesn't exist"""
        self.logger.info("Starting generate-mp3-new command")
        
        # Get all folders in audiobooks bucket
        folders = self.list_audiobook_folders()
        if not folders:
            self.logger.warning("No folders found in audiobooks bucket")
            return
        
        successful = 0
        failed = 0
        skipped = 0
        total_folders = len(folders)
        
        self.logger.info(f"Checking {total_folders} folders for MP3 conversion...")
        
        for i, slug in enumerate(folders, 1):
            self.logger.info(f"Processing folder {i}/{total_folders}: {slug}")
            
            # Check if MP3 already exists
            if self.check_mp3_exists(slug):
                self.logger.info(f"MP3 already exists for {slug}, skipping")
                skipped += 1
                continue
            
            # Check if M4B exists
            if not self.check_m4b_exists(slug):
                self.logger.warning(f"No M4B file found for {slug}, skipping")
                skipped += 1
                continue
            
            # Process this folder: download M4B, convert to MP3, upload MP3, update database
            m4b_path = None
            mp3_path = None
            
            try:
                # Step 1: Download M4B
                self.logger.info(f"Downloading M4B for {slug}")
                m4b_path = self.download_m4b_from_bucket(slug)
                if not m4b_path:
                    self.logger.error(f"Failed to download M4B for {slug}")
                    failed += 1
                    continue
                
                # Step 2: Convert M4B to MP3
                self.logger.info(f"Converting M4B to MP3 for {slug}")
                mp3_path = self.convert_m4b_to_mp3(m4b_path)
                if not mp3_path:
                    self.logger.error(f"Failed to convert M4B to MP3 for {slug}")
                    failed += 1
                    continue
                
                # Step 3: Upload MP3 to bucket
                self.logger.info(f"Uploading MP3 for {slug}")
                mp3_url = self.upload_mp3_to_bucket(slug, mp3_path)
                if not mp3_url:
                    self.logger.error(f"Failed to upload MP3 for {slug}")
                    failed += 1
                    continue
                
                # Step 4: Update database
                self.logger.info(f"Updating database MP3 URL for {slug}")
                if not self.update_audiobook_mp3_url(slug, mp3_url):
                    self.logger.error(f"Failed to update database MP3 URL for {slug}")
                    failed += 1
                    continue
                
                successful += 1
                self.logger.info(f"✅ {slug} - MP3 generated successfully ({successful}/{total_folders})")
                
            except Exception as e:
                self.logger.error(f"Error processing MP3 for {slug}: {e}")
                failed += 1
                
            finally:
                # Clean up temporary files
                if m4b_path and m4b_path.exists():
                    m4b_path.unlink()
                    self.logger.info(f"Cleaned up {m4b_path}")
                if mp3_path and mp3_path.exists():
                    mp3_path.unlink()
                    self.logger.info(f"Cleaned up {mp3_path}")
        
        self.logger.info(f"Generate-mp3-new completed: {successful} successful, {failed} failed, {skipped} skipped")
        
    def cmd_generate_single(self, slug: str):
        """Command: generate <slug> - Process a single book by slug"""
        self.logger.info(f"Starting generate command for slug: {slug}")
        
        if self.process_single_book(slug):
            self.logger.info(f"✅ Successfully generated audiobook for: {slug}")
        else:
            self.logger.error(f"❌ Failed to generate audiobook for: {slug}")
            sys.exit(1)


def main():
    """Main entry point with command routing"""
    parser = argparse.ArgumentParser(description='Audiobook Pipeline Management')
    parser.add_argument('command', choices=['regenerate-all', 'generate-new', 'generate', 'generate-mp3-new'], 
                       help='Command to run')
    parser.add_argument('slug', nargs='?', help='Slug for generate command')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.command == 'generate' and not args.slug:
        print("Error: 'generate' command requires a slug argument")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = AudiobookPipeline()
    
    # Route to appropriate command handler
    try:
        if args.command == 'regenerate-all':
            pipeline.cmd_regenerate_all()
        elif args.command == 'generate-new':
            pipeline.cmd_generate_new()
        elif args.command == 'generate':
            pipeline.cmd_generate_single(args.slug)
        elif args.command == 'generate-mp3-new':
            pipeline.cmd_generate_mp3_new()
    except KeyboardInterrupt:
        pipeline.logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        pipeline.logger.error(f"Unexpected error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()