#!/usr/bin/env python
"""
Launcher script for DAIA demos
Provides easy access to both Gradio and Streamlit interfaces
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Launch DAIA demo interface")
    parser.add_argument(
        '--interface',
        type=str,
        choices=['gradio', 'streamlit', 'both'],
        default='streamlit',
        help='Which interface to launch (default: streamlit)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DAIA - Demo Launcher")
    print("="*60)
    print()
    
    if args.interface == 'gradio' or args.interface == 'both':
        print("🚀 Launching Gradio interface...")
        print("   URL: http://localhost:7860")
        print()
        os.system("python src/demo.py")
    
    if args.interface == 'streamlit':
        print("🚀 Launching Streamlit interface...")
        print("   URL: http://localhost:8501")
        print()
        print("Features:")
        print("  ✓ Image upload and analysis")
        print("  ✓ Model statistics dashboard")
        print("  ✓ Training curves visualization")
        print("  ✓ Risk assessment")
        print("  ✓ XAI explanations")
        print()
        os.system("streamlit run src/app.py")


if __name__ == "__main__":
    main()
