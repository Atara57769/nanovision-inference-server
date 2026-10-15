import path from 'node:path'
import { fileURLToPath } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const srcDir = fileURLToPath(new URL('./src', import.meta.url))

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@core': path.resolve(srcDir, 'core'),
      '@infrastructure': path.resolve(srcDir, 'infrastructure'),
      '@ui': path.resolve(srcDir, 'ui'),
      '@app': path.resolve(srcDir, 'app'),
    },
  },
})
