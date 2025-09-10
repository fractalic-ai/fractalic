# Fractalic.ai Documentation Setup Guide

## Overview
This guide explains how to configure your `fractalic-site` repository to serve documentation at `fractalic.ai/docs`.

## How the Deployment Works

1. **Trigger**: When you push changes to `/docs` folder in the `fractalic` repository
2. **Build**: GitHub Actions builds VitePress documentation
3. **Deploy**: Built files are pushed to `fractalic-site/docs/` folder
4. **Serve**: Your main site serves them at `fractalic.ai/docs`

## Required Configuration in fractalic-site

### Option 1: Simple Redirect (Recommended)
Add this to your main `index.html` in `fractalic-site`:

```html
<!-- Add this script to handle /docs routing -->
<script>
  // Handle /docs routing
  if (window.location.pathname.startsWith('/docs')) {
    // The docs are already in the correct location
    // GitHub Pages will serve them automatically
  }
</script>
```

### Option 2: Server Configuration
If you have server access, add this to your server config:

**Nginx:**
```nginx
location /docs/ {
    try_files $uri $uri/ /docs/index.html;
}
```

**Apache (.htaccess in root):**
```apache
RewriteEngine On
RewriteRule ^docs/(.*)$ docs/$1 [L]
```

### Option 3: GitHub Pages Automatic Serving
Since your site is on GitHub Pages, it will automatically serve files from the `/docs` folder at `/docs/` URL path. No additional configuration needed!

## Verification Steps

1. **Check Repository**: After deployment, verify files exist in `fractalic-site/docs/`
2. **Test URL**: Visit `https://fractalic.ai/docs/` 
3. **Check Build**: Monitor GitHub Actions in both repositories

## Troubleshooting

### 404 Errors
- Ensure files are in `fractalic-site/docs/` folder
- Check that GitHub Pages is enabled for `fractalic-site`
- Verify the base path in VitePress config is `/docs/`

### Build Failures
- Check Node.js version compatibility
- Verify all markdown links are valid
- Ensure VitePress dependencies are up to date

### Deployment Issues
- Verify `DEPLOY_TOKEN` has correct permissions
- Check repository access rights
- Review GitHub Actions logs

## DNS Configuration (Namecheap)

Your current DNS should work as-is since you're using GitHub Pages. If you need to make changes:

1. **A Records** (if using apex domain):
   ```
   185.199.108.153
   185.199.109.153
   185.199.110.153
   185.199.111.153
   ```

2. **CNAME** (if using www subdomain):
   ```
   fractalic-ai.github.io
   ```

## Testing the Setup

1. Make a small change to any file in `/docs` folder
2. Push to `main` branch
3. Check GitHub Actions tab for workflow progress
4. Visit `https://fractalic.ai/docs/` after deployment completes

The documentation should be live and automatically update whenever you modify files in the `/docs` folder!
