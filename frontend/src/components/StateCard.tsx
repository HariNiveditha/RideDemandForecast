import type { ReactNode } from 'react'
import { TaxiIllustration } from './TaxiIcons'

type StateCardProps = {
  tone?: 'loading' | 'error' | 'empty'
  title: string
  description?: string
  compact?: boolean
  children?: ReactNode
}

function StateCard({ tone = 'empty', title, description, compact = false, children }: StateCardProps) {
  return (
    <div
      className={`state-card tone-${tone} ${compact ? 'is-compact' : ''}`}
      role={tone === 'error' ? 'alert' : 'status'}
      aria-live="polite"
    >
      <div className="state-visual">
        <TaxiIllustration width={compact ? 112 : 150} motion={tone === 'loading'} />
        <span className="state-road" aria-hidden="true" />
      </div>
      <div className="state-copy">
        <strong>{title}</strong>
        {description && <span>{description}</span>}
        {children}
      </div>
    </div>
  )
}

export default StateCard
