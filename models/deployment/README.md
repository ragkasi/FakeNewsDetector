# Deployment Model Files

This directory contains the deployment-ready model files for the Fake News Detector.

## Required Files

The following files are required for deployment:

- ✅ `config.json` - Model configuration
- ✅ `tokenizer_config.json` - Tokenizer configuration  
- ✅ `vocab.txt` - Vocabulary file
- ✅ `special_tokens_map.json` - Special tokens mapping
- ⚠️ `pytorch_model.bin` - Model weights (418MB - too large for Git)

## Model Weights

The `pytorch_model.bin` file is too large for GitHub (418MB). 

### For Local Development:
Run the conversion script to generate it:
```bash
python scripts/convert_safetensors_to_pytorch.py
```

### For Production Deployment:
Consider these options:

1. **Git LFS** (Git Large File Storage):
   ```bash
   git lfs track "*.bin"
   git add .gitattributes
   git add models/deployment/bert-fake-news/pytorch_model.bin
   git commit -m "Add model weights with Git LFS"
   ```

2. **Model Registry** (Recommended for production):
   - Upload to Hugging Face Model Hub
   - Use cloud storage (AWS S3, Google Cloud Storage)
   - Download during deployment initialization

3. **Alternative Format**:
   - Use `model.safetensors` instead (already available)
   - Update app.py to load safetensors format

## Model Performance

- **Checkpoint**: checkpoint-5000 (best balance)
- **Training Data**: Combined datasets (fake-and-real-news, LIAR)
- **Known Bias**: Tends toward real news classification
- **Confidence Levels**: >95% "Likely", 75-95% "Probably", <75% "Leaning"

## Usage

The model is automatically loaded by the Streamlit app when placed in this directory structure. 