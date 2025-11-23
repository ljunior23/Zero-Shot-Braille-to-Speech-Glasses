"""
Generate audio for sentences using free Google Translate TTS.
Zero cost - uses gTTS library.
"""

from gtts import gTTS
from pathlib import Path
from tqdm import tqdm
import argparse
import time
import hashlib


def text_to_speech(text: str, output_path: str, lang: str = 'en', slow: bool = False) -> bool:
    """
    Convert text to speech using Google Translate TTS.
    
    Args:
        text: Text to convert
        output_path: Where to save MP3 file
        lang: Language code
        slow: Whether to speak slowly
        
    Returns:
        Success boolean
    """
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"Error generating TTS for '{text[:50]}...': {e}")
        return False


def generate_filename(text: str) -> str:
    """Generate consistent filename from text hash."""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"audio_{text_hash}.mp3"


def main():
    """Generate audio for all sentences."""
    parser = argparse.ArgumentParser(description='Generate TTS audio from text')
    parser.add_argument('--input', default='data/text/sentences.txt',
                       help='Input sentences file')
    parser.add_argument('--output-dir', default='data/audio',
                       help='Output directory for audio files')
    parser.add_argument('--lang', default='en',
                       help='Language code')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Process in batches (for rate limiting)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sentences
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        print("Run: python data_collection/download_text.py first")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    print("="*60)
    print("Generating Free TTS Audio (Google Translate)")
    print("="*60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Sentences: {len(sentences):,}")
    print(f"Language: {args.lang}")
    print()
    
    # Generate audio for each sentence
    successful = 0
    failed = 0
    skipped = 0
    
    # Create metadata file
    metadata = []
    
    pbar = tqdm(sentences, desc="Generating audio")
    for idx, sentence in enumerate(pbar):
        # Generate filename
        filename = generate_filename(sentence)
        output_path = output_dir / filename
        
        # Skip if already exists
        if output_path.exists():
            skipped += 1
            metadata.append({
                'id': idx,
                'filename': filename,
                'text': sentence,
                'duration': None  # Would need to load audio to get duration
            })
            continue
        
        # Generate audio
        success = text_to_speech(sentence, str(output_path), lang=args.lang)
        
        if success:
            successful += 1
            metadata.append({
                'id': idx,
                'filename': filename,
                'text': sentence,
                'duration': None
            })
        else:
            failed += 1
        
        # Update progress
        pbar.set_postfix({
            'success': successful,
            'failed': failed,
            'skipped': skipped
        })
        
        # Rate limiting
        if (idx + 1) % args.batch_size == 0:
            print(f"\n  Processed {idx + 1} sentences, pausing briefly...")
            time.sleep(2)
        else:
            time.sleep(args.delay)
    
    # Save metadata
    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n" + "="*60)
    print("Summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Total audio files: {successful + skipped}")
    print(f"\n✓ Metadata saved to: {metadata_path}")
    print(f"\n✓ Audio generation complete!")


if __name__ == "__main__":
    main()