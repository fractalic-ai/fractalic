import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Fractalic Documentation',
  description: 'AI workflows in Markdown + YAML operation blocks',
  base: process.env.NODE_ENV === 'production' ? '/docs/' : '/',
  lang: 'en-US',
  lastUpdated: true,
  
  head: [
    ['meta', { name: 'theme-color', content: '#3c82f6' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en' }],
    ['meta', { name: 'og:title', content: 'Fractalic Documentation' }],
    ['meta', { name: 'og:description', content: 'AI workflows in Markdown + YAML operation blocks' }],
    ['meta', { name: 'og:site_name', content: 'Fractalic' }],
  ],
  
  themeConfig: {
    siteTitle: 'Fractalic Docs',
    
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Quick Start', link: '/quick-start' },
      { 
        text: 'Guide',
        items: [
          { text: 'Core Concepts', link: '/core-concepts' },
          { text: 'Syntax Reference', link: '/syntax-reference' },
          { text: 'Operations Reference', link: '/operations-reference' }
        ]
      },
      { 
        text: 'Advanced',
        items: [
          { text: 'Advanced LLM Features', link: '/advanced-llm-features' },
          { text: 'Context Management', link: '/context-management' },
          { text: 'Agent Workflows', link: '/agent-modular-workflows' }
        ]
      }
    ],

    sidebar: {
      '/': [
        {
          text: 'Getting Started',
          collapsible: false,
          items: [
            { text: 'Introduction', link: '/introduction' },
            { text: 'Quick Start', link: '/quick-start' },
            { text: 'Core Concepts', link: '/core-concepts' }
          ]
        },
        {
          text: 'Reference',
          collapsible: false,
          items: [
            { text: 'Syntax Reference', link: '/syntax-reference' },
            { 
              text: 'Operations Reference', 
              link: '/operations-reference',
              items: [
                { text: '@import', link: '/operations-reference#import' },
                { text: '@llm', link: '/operations-reference#llm' },
                { text: '@shell', link: '/operations-reference#shell' },
                { text: '@run', link: '/operations-reference#run' },
                { text: '@return', link: '/operations-reference#return' },
                { text: '@goto', link: '/operations-reference#goto-experimental' }
              ]
            },
            { text: 'Configuration', link: '/configuration' }
          ]
        },
        {
          text: 'Advanced Features',
          collapsible: false,
          items: [
            { text: 'Advanced LLM Features', link: '/advanced-llm-features' },
            { text: 'Context Management', link: '/context-management' },
            { text: 'File & Import Semantics', link: '/file-import-semantics' },
            { text: 'Custom Tools', link: '/custom-tools' }
          ]
        },
        {
          text: 'Integration & Workflows',
          collapsible: false,
          items: [
            { text: 'Shell Integration Patterns', link: '/shell-integration-patterns' },
            { text: 'Agent & Modular Workflows', link: '/agent-modular-workflows' },
            { text: 'MCP Integration', link: '/mcp-integration' },
            { text: 'Git-Backed Sessions', link: '/git-backed-sessions' }
          ]
        },
        {
          text: 'Deployment & API',
          collapsible: false,
          items: [
            { text: 'UI Server & API', link: '/ui-server-api' },
            { text: 'Publisher & Deployment', link: '/publisher-deployment' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/fractalic-ai/fractalic' }
    ],

    search: {
      provider: 'local'
    },

    editLink: {
      pattern: 'https://github.com/fractalic-ai/fractalic/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2025 Fractalic AI'
    }
  },

  vite: {
    server: {
      fs: {
        allow: ['..']
      }
    }
  },

  cleanUrls: true
})
