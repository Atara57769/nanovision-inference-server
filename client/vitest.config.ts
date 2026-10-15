import { mergeConfig } from 'vite'
import { defineConfig as defineVitestConfig } from 'vitest/config'

import viteConfig from './vite.config.ts'

export default mergeConfig(
  viteConfig,
  defineVitestConfig({
    test: {
      globals: true,
      projects: [
        {
          extends: true,
          test: {
            name: 'core',
            // core/ has zero framework dependencies — must stay testable
            // in plain Node, no DOM.
            environment: 'node',
            include: ['src/core/**/*.{test,spec}.ts'],
          },
        },
        {
          extends: true,
          test: {
            name: 'ui',
            // ui/ renders React components and needs a DOM.
            environment: 'jsdom',
            include: ['src/ui/**/*.{test,spec}.{ts,tsx}'],
            setupFiles: ['./src/ui/test/setup.ts'],
          },
        },
        {
          extends: true,
          test: {
            name: 'infrastructure',
            // infrastructure/ decodes PNGs and calls fetch — Node-testable
            // by design (no <canvas>/DOM dependency).
            environment: 'node',
            include: ['src/infrastructure/**/*.{test,spec}.ts'],
          },
        },
      ],
    },
  }),
)
