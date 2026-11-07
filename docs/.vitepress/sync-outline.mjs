import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync } from 'node:fs'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { chapterHeading, chapters, parts } from './outline.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = join(__dirname, '..', '..')
const writeStubs = process.argv.includes('--write-stubs')
const checkOnly = process.argv.includes('--check')
const changedFiles = []

const alphaNotice = '::: danger ⚠️ Alpha内测版本警告\n此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。\n:::'

async function writeIfChanged(path, content) {
  const current = existsSync(path) ? await readFile(path, 'utf8') : null
  if (current === content) return

  changedFiles.push(path)
  if (!checkOnly) {
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, content)
  }
}

function frontmatter(chapter) {
  const heading = chapterHeading(chapter)
  return `---\ntitle: "${heading}"\ndescription: "Hello MLSys · ${heading} · ${chapter.summary}"\n---\n`
}

function chapterMarkdown(chapter) {
  const sections = chapter.sections.map(([title, body], index) => {
    const number = chapter.number ? `${chapter.number}.${index + 1} ` : ''
    return `## ${number}${title}\n\n${body}`
  })

  const blocks = [
    frontmatter(chapter).trimEnd(),
    `# ${chapterHeading(chapter)}`,
    alphaNotice,
    `## 本章导读\n\n> ${chapter.lead}`,
    ...sections,
  ]

  if (chapter.code) {
    blocks.push(`## 本章小结\n\n- 本章目前是 Alpha 阶段的大纲骨架，正式正文会在对应实验跑通后补齐。\n- 与本章相关的代码、实验配置与测量记录会放在 \`${chapter.code}/\`。`)
  }

  blocks.push('## 延伸阅读\n\n- 待补充。')
  return `${blocks.join('\n\n')}\n`
}

// 已有正文只同步 frontmatter 与一级标题，不覆盖作者写下的内容。
function syncChapterMetadata(existing, chapter) {
  const body = existing.replace(/^---\n[\s\S]*?\n---\n/, '').replace(/^\n+/, '')
  return `${frontmatter(chapter)}\n${body}`.replace(/^# .+$/m, `# ${chapterHeading(chapter)}`)
}

async function syncChapters() {
  for (const chapter of chapters) {
    const path = join(repoRoot, chapter.source)
    if (!existsSync(path) || writeStubs) {
      await writeIfChanged(path, chapterMarkdown(chapter))
    } else {
      await writeIfChanged(path, syncChapterMetadata(await readFile(path, 'utf8'), chapter))
    }
  }
}

function readmeDirectory() {
  const lines = [
    '## 目录',
    '',
    '> 章节由 `docs/.vitepress/outline.mjs` 统一维护，修改后运行 `npm run docs:sync-outline` 即可同步 README 目录、站点导航与章节骨架。',
    '',
    '| 章节名 | 简介 | 状态 |',
    '| ---- | ---- | ---- |',
  ]

  for (const part of parts) {
    lines.push(`| **${part.readmeTitle}** | | |`)
    for (const chapter of chapters.filter((item) => item.part === part)) {
      lines.push(`| [${chapterHeading(chapter)}](./${chapter.source}) | ${chapter.summary} | ${chapter.status} |`)
    }
  }

  return `${lines.join('\n')}\n`
}

async function syncReadme() {
  const path = join(repoRoot, 'README.md')
  const current = await readFile(path, 'utf8')
  const next = current.replace(/## 目录\n[\s\S]*?\n## 贡献者名单/, `${readmeDirectory()}\n## 贡献者名单`)
  await writeIfChanged(path, next)
}

async function syncCurriculum() {
  const lines = [
    '---',
    'title: 全书目录',
    'description: Hello MLSys 的学习路线、全部章节与附录。',
    'outline: [2, 2]',
    'editLink: false',
    'prev: false',
    'next: false',
    '---',
    '',
    '<!-- 此页由 docs/.vitepress/sync-outline.mjs 生成，请在 outline.mjs 中维护章节。 -->',
    '',
    '# 全书目录',
    '',
    '从入门开始，依次学习建模与校准，再完成与 AI 一起设计的综合实践。选修案例和附录可按需查阅。',
    '',
    '::: info 阅读说明',
    '教程处于 Alpha 阶段，部分章节仍为大纲或待补充内容，请以各章现有正文与验证记录为准。',
    ':::',
  ]

  for (const part of parts) {
    lines.push('', `## ${part.readmeTitle}`, '')
    for (const chapter of chapters.filter((item) => item.part === part)) {
      lines.push(`- [${chapterHeading(chapter)}](${chapter.path}) — ${chapter.summary}`)
    }
  }

  await writeIfChanged(join(repoRoot, 'docs/curriculum/index.md'), `${lines.join('\n')}\n`)
}

async function main() {
  await syncChapters()
  await syncReadme()
  await syncCurriculum()

  if (checkOnly && changedFiles.length > 0) {
    console.error('Docs outline is out of sync:')
    for (const path of changedFiles) console.error(`- ${path}`)
    process.exit(1)
  }

  console.log(changedFiles.length > 0
    ? `Synced docs outline (${changedFiles.length} file${changedFiles.length === 1 ? '' : 's'}).`
    : 'Docs outline already in sync.')
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
