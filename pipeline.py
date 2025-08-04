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
                
            self.logger.info(f"Successfully processed book: {slug}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing book {slug}: {e}")
            return False
            
        finally:
            # Clean up temporary files
            if epub_path or m4b_path:
                self.cleanup_temp_files(epub_path, m4b_path)
                
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
    parser.add_argument('command', choices=['regenerate-all', 'generate-new', 'generate'], 
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
    except KeyboardInterrupt:
        pipeline.logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        pipeline.logger.error(f"Unexpected error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()