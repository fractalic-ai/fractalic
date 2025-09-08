import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Fractalic Documentation',
  description: 'AI workflows in Markdown + YAML operation blocks',
  
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Quick Start', link: '/quick-start' }
    ],
    
    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Introduction', link: '/introduction' },
          { text: 'Quick Start', link: '/quick-start' }
        ]
      }
    ]
  }
})
