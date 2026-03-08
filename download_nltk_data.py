import nltk

print("Downloading NLTK data packages...")

# Download required packages
packages = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']

for package in packages:
    try:
        nltk.download(package)
        print(f"✓ Downloaded {package}")
    except Exception as e:
        print(f"✗ Failed to download {package}: {e}")

print("\nAll NLTK data packages downloaded successfully!")