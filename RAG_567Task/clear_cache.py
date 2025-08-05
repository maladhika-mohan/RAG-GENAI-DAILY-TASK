#!/usr/bin/env python3
"""
Clear Streamlit Cache Script
============================

This script clears Streamlit cache and temporary files to resolve
common issues like duplicate element IDs or stale session state.

Usage:
    python clear_cache.py

What it does:
1. Clears Streamlit cache directory
2. Removes temporary session files
3. Resets any cached models or data
4. Provides instructions for manual cleanup if needed
"""

import os
import shutil
import sys
from pathlib import Path

def clear_streamlit_cache():
    """Clear Streamlit cache directories"""
    
    print("🧹 RAG System Cache Cleaner")
    print("=" * 40)
    
    # Common Streamlit cache locations
    cache_locations = [
        Path.home() / ".streamlit",
        Path.cwd() / ".streamlit",
        Path.home() / ".cache" / "streamlit",
        Path("/tmp") / "streamlit" if os.name != 'nt' else Path(os.environ.get('TEMP', '')) / "streamlit"
    ]
    
    cleared_count = 0
    
    for cache_path in cache_locations:
        if cache_path.exists():
            try:
                print(f"🗑️  Clearing: {cache_path}")
                shutil.rmtree(cache_path)
                cleared_count += 1
                print(f"✅ Cleared: {cache_path}")
            except Exception as e:
                print(f"⚠️  Could not clear {cache_path}: {e}")
    
    # Clear local data directories
    local_dirs = ['data/temp', 'data/cache']
    for dir_path in local_dirs:
        path = Path(dir_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Cleared local directory: {dir_path}")
                cleared_count += 1
            except Exception as e:
                print(f"⚠️  Could not clear {dir_path}: {e}")
    
    print(f"\n🎉 Cache cleanup complete! Cleared {cleared_count} locations.")
    
    # Instructions
    print("\n📋 Next Steps:")
    print("1. Restart the Streamlit application:")
    print("   streamlit run app.py")
    print("\n2. If you still see duplicate element errors:")
    print("   - Close all browser tabs with the app")
    print("   - Wait 10 seconds")
    print("   - Restart the app")
    print("   - Open a new browser tab")
    
    print("\n3. For persistent issues:")
    print("   - Use incognito/private browsing mode")
    print("   - Try a different port: streamlit run app.py --server.port 8502")

def clear_python_cache():
    """Clear Python __pycache__ directories"""
    
    print("\n🐍 Clearing Python cache...")
    
    cache_count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            cache_path = Path(root) / '__pycache__'
            try:
                shutil.rmtree(cache_path)
                cache_count += 1
                print(f"✅ Cleared: {cache_path}")
            except Exception as e:
                print(f"⚠️  Could not clear {cache_path}: {e}")
    
    print(f"🎉 Cleared {cache_count} Python cache directories.")

def main():
    """Main function"""
    
    try:
        clear_streamlit_cache()
        clear_python_cache()
        
        print("\n" + "=" * 50)
        print("🚀 Cache cleanup complete!")
        print("Your RAG system should now run without cache issues.")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n❌ Cache cleanup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
