import js from '@eslint/js'
import boundaries from 'eslint-plugin-boundaries'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
      boundaries,
    },
    settings: {
      'boundaries/elements': [
        { type: 'core', pattern: 'src/core/**' },
        { type: 'infrastructure', pattern: 'src/infrastructure/**' },
        { type: 'ui', pattern: 'src/ui/**' },
        { type: 'app', pattern: 'src/app/**' },
      ],
      // boundaries/dependencies can't classify an import it can't resolve —
      // this teaches it to follow @core/* aliases and extensionless .ts imports.
      'import/resolver': {
        typescript: true,
      },
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // Ports & Adapters boundaries: core has zero outward deps, ui/
      // infrastructure both depend only on core, app wires everything.
      'boundaries/dependencies': [
        'error',
        {
          default: 'allow',
          // Without this, the rule silently skips every dependency whose
          // target isn't a local project file — so the react/react-dom
          // policy below would never even run.
          checkAllOrigins: true,
          policies: [
            {
              from: { element: { type: 'core' } },
              disallow: {
                to: { element: { type: ['infrastructure', 'ui', 'app'] } },
              },
              message:
                'core/ must have zero dependencies on infrastructure, ui, or app.',
            },
            {
              from: { element: { type: 'core' } },
              disallow: {
                to: {
                  module: {
                    origin: 'external',
                    source: ['react', 'react-dom'],
                  },
                },
              },
              message: 'core/ must stay framework-free (no react imports).',
            },
            {
              from: { element: { type: 'infrastructure' } },
              disallow: { to: { element: { type: ['ui', 'app'] } } },
              message:
                'infrastructure/ may only depend on core (the ports it implements).',
            },
            {
              from: { element: { type: 'ui' } },
              disallow: { to: { element: { type: ['infrastructure', 'app'] } } },
              message:
                'ui/ must depend on core ports only — never call infrastructure directly.',
            },
          ],
        },
      ],
    },
  },
)
