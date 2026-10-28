import { defineConfig } from 'vitepress'
import { navItems, repoUrl, sidebar } from './outline.mjs'

export default defineConfig({
  lang: 'zh-CN',
  title: 'Hello MLSys',
  description: '用模拟器预测驱动 LLM 推理系统设计，并在 AMD GPU 上实测验证与校准的实战教程',
  base: '/hello-mlsys/',

  cleanUrls: true,

  themeConfig: {
    nav: navItems,

    sidebar,

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档',
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
            },
          },
        },
      },
    },

    socialLinks: [
      { icon: 'github', link: repoUrl },
    ],

    editLink: {
      pattern: `${repoUrl}/blob/main/docs/:path`,
      text: '在 GitHub 上编辑此页',
    },

    outline: {
      level: [2, 3],
      label: '本章目录',
    },

    docFooter: {
      prev: '上一章',
      next: '下一章',
    },

    footer: {
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可',
    },
  },
})
