import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from '@ui/App'

import { createContainer } from './container'
import './index.css'

const container = createContainer()
const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Missing #root element in index.html')
}

createRoot(rootElement).render(
  <StrictMode>
    <App
      predictionGateway={container.predictionGateway}
      healthGateway={container.healthGateway}
    />
  </StrictMode>,
)
