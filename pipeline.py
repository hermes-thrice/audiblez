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
            response = self.supabase.storage.from_("audiobooks").upload(
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
            
            # Use ffmpeg to convert M4B to MP3 with optimized settings for smaller file size
            command = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', str(m4b_path),  # Input M4B file
                '-c:a', 'mp3',  # Convert to MP3
                '-b:a', '64k',  # Lower bitrate for smaller files (was 128k)
                '-ar', '22050',  # Lower sample rate for speech (was default 44100)
                '-ac', '1',  # Mono audio for speech (was stereo)
                '-q:a', '4',  # Quality setting (0-9, 4 is good balance)
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
                self.logger.info(f"Successfully converted to MP3: {mp3_path}")
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
            
    def generate_audiobook(self, epub_path: Path) -> Optional[Path]:
        """Run audiobook generation command and return path to generated m4b"""
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
                    return m4b_path
                else:
                    self.logger.error(f"Command succeeded but m4b file not found: {m4b_path}")
                    return None
            else:
                self.logger.error(f"Audiobook generation failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Audiobook generation timed out for {epub_path}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to generate audiobook for {epub_path}: {e}")
            return None
            
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
            m4b_path = self.generate_audiobook(epub_path)
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