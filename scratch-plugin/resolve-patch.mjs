import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

// Patch-relative names are resolved by the loader against the profile directory.
const patch = readFileSync(new URL('./src/cordis.yml', import.meta.url), 'utf8')
process.stdout.write(patch.replace(/^(\s+name: )'\.\/([^']+)'$/gm, (_, prefix, name) =>
  prefix + JSON.stringify(fileURLToPath(new URL(`./src/${name}`, import.meta.url)))))
