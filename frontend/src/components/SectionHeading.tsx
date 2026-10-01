import type { ReactNode } from 'react'
import { TaxiIllustration } from './TaxiIcons'

type SectionHeadingProps = {
  eyebrow: string
  title: string
  showTaxi?: boolean
  children?: ReactNode
}

function SectionHeading({ eyebrow, title, showTaxi = true, children }: SectionHeadingProps) {
  return (
    <div className="section-heading">
      <div className="section-title">
        {showTaxi && (
          <span className="section-taxi" aria-hidden="true">
            <TaxiIllustration width={46} />
          </span>
        )}
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
      </div>
      {children}
    </div>
  )
}

export default SectionHeading
