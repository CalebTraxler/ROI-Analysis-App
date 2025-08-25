# 🚀 Streamlit Community Cloud Deployment Guide

## Quick Deploy to Streamlit Community Cloud

### 1. **Fork the Repository**
- Go to [https://github.com/CalebTraxler/ROI-Analysis-App](https://github.com/CalebTraxler/ROI-Analysis-App)
- Click "Fork" to create your own copy

### 2. **Deploy to Streamlit Community Cloud**
- Visit [https://share.streamlit.io/](https://share.streamlit.io/)
- Sign in with your GitHub account
- Click "New app"
- Select your forked repository
- Set the main file path to: `ROI_optimized.py`
- Click "Deploy!"

### 3. **Configuration**
The app is pre-configured with:
- ✅ `.streamlit/config.toml` - Production settings
- ✅ `requirements.txt` - All dependencies
- ✅ `ROI_optimized.py` - Main application file
- ✅ Performance optimizations enabled

### 4. **Performance Features**
- **Multi-layer caching system**
- **Parallel geocoding**
- **Pre-processed data cache**
- **85%+ performance improvement**

### 5. **First Run**
- Initial deployment: 2-3 minutes
- First user visit: 8-10 seconds (building cache)
- Subsequent visits: 2-3 seconds (cached)

## 🎯 Deployment Checklist

- [ ] Repository forked to your account
- [ ] Streamlit Community Cloud account created
- [ ] App deployed with `ROI_optimized.py` as main file
- [ ] All dependencies installed successfully
- [ ] App loads without errors
- [ ] Performance metrics showing improvements

## 🔧 Troubleshooting

### Common Issues:
1. **Import Errors**: Check `requirements.txt` has all dependencies
2. **Slow Loading**: First run builds cache - subsequent runs are fast
3. **Memory Issues**: App is optimized for Streamlit Community Cloud limits

### Support:
- Check Streamlit Community Cloud logs
- Verify GitHub repository settings
- Ensure main file path is correct

---

**Your optimized ROI app will be live at: `https://your-app-name.streamlit.app`**
