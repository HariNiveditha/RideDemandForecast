import { useId } from 'react'

export const CAB_YELLOW = '#F6BE1A'
export const ASPHALT = '#16181D'
const GLASS = '#223445'

type TaxiBadge = 'pin' | 'peak' | 'clock'

type TaxiIconProps = {
  size?: number
  badge?: TaxiBadge
  className?: string
  title?: string
}

const smallChecker = Array.from({ length: 12 }, (_, index) => index)

function BadgeGlyph({ badge }: { badge: TaxiBadge }) {
  return (
    <g>
      <circle cx="19.4" cy="5.4" r="4.3" fill="#10263F" stroke="#fff" strokeWidth="1" />
      {badge === 'pin' && (
        <>
          <path d="M19.4 2.9a1.9 1.9 0 0 0-1.9 1.9c0 1.4 1.9 3.4 1.9 3.4s1.9-2 1.9-3.4a1.9 1.9 0 0 0-1.9-1.9Z" fill={CAB_YELLOW} />
          <circle cx="19.4" cy="4.8" r=".7" fill="#10263F" />
        </>
      )}
      {badge === 'peak' && (
        <>
          <path d="M17.1 7.3 18.7 5.5l1.1 1.1 1.9-2.3" fill="none" stroke={CAB_YELLOW} strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M20.2 4.1h1.6v1.6" fill="none" stroke={CAB_YELLOW} strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
        </>
      )}
      {badge === 'clock' && (
        <>
          <circle cx="19.4" cy="5.4" r="2.3" fill="none" stroke={CAB_YELLOW} strokeWidth="1" />
          <path d="M19.4 4.2v1.3l.9.6" fill="none" stroke={CAB_YELLOW} strokeWidth=".9" strokeLinecap="round" />
        </>
      )}
    </g>
  )
}

export function TaxiIcon({ size = 20, badge, className, title }: TaxiIconProps) {
  const isLabelled = Boolean(title)

  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      className={className}
      role={isLabelled ? 'img' : undefined}
      aria-label={title}
      aria-hidden={isLabelled ? undefined : true}
      focusable="false"
    >
      <rect x="10.3" y="4.7" width="3.4" height="2.2" rx=".6" fill="#fff" stroke={ASPHALT} strokeWidth="1" />
      <path
        d="M4.2 17.2Q3 17.2 3 16v-2.8q0-1.6 1.6-1.9l2.5-.4 2-3.2q.4-.6 1.2-.6h3.4q.8 0 1.2.6l2 3.2 2.5.4q1.6.3 1.6 1.9V16q0 1.2-1.2 1.2Z"
        fill={CAB_YELLOW}
        stroke={ASPHALT}
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path d="M9 10.9 10.4 8.6h1.1v2.3Zm3.5-2.3h1.1l1.4 2.3h-2.5Z" fill={GLASS} />
      {smallChecker.map((index) => (
        <rect
          key={index}
          x={4.4 + index * 1.28}
          y={index % 2 === 0 ? 12.7 : 13.98}
          width="1.28"
          height="1.28"
          fill={ASPHALT}
        />
      ))}
      <circle cx="7.6" cy="17.2" r="2.1" fill={ASPHALT} />
      <circle cx="7.6" cy="17.2" r=".8" fill="#D5DCE2" />
      <circle cx="16.4" cy="17.2" r="2.1" fill={ASPHALT} />
      <circle cx="16.4" cy="17.2" r=".8" fill="#D5DCE2" />
      {badge && <BadgeGlyph badge={badge} />}
    </svg>
  )
}

type TaxiIllustrationProps = {
  width?: number
  motion?: boolean
  className?: string
  title?: string
}

export function TaxiIllustration({ width = 160, motion = false, className, title }: TaxiIllustrationProps) {
  const patternId = useId()
  const isLabelled = Boolean(title)

  return (
    <svg
      viewBox="0 0 120 72"
      width={width}
      height={(width * 72) / 120}
      className={className}
      role={isLabelled ? 'img' : undefined}
      aria-label={title}
      aria-hidden={isLabelled ? undefined : true}
      focusable="false"
    >
      <defs>
        <pattern id={patternId} width="6" height="6" patternUnits="userSpaceOnUse">
          <rect width="3" height="3" fill={ASPHALT} />
          <rect x="3" y="3" width="3" height="3" fill={ASPHALT} />
        </pattern>
      </defs>

      <ellipse cx="60" cy="66" rx="48" ry="3" fill={ASPHALT} opacity=".14" />

      {motion && (
        <g className="taxi-speed" stroke={ASPHALT} strokeWidth="2" strokeLinecap="round" opacity=".35">
          <path d="M1 30h6" />
          <path d="M0 39h7" />
          <path d="M2 48h5" />
        </g>
      )}

      <g className="taxi-body">
        <rect x="50" y="10.5" width="20" height="8" rx="2" fill="#fff" stroke={ASPHALT} strokeWidth="2" />
        <text x="60" y="16.7" textAnchor="middle" fontSize="5.2" fontWeight="800" letterSpacing=".6" fill={ASPHALT} fontFamily="inherit">
          TAXI
        </text>

        <path
          d="M14 56q-4 0-4-4V42q0-6 6-7l14-2 9-12q2-3 6-3h30q4 0 6 3l10 12 13 2q6 1 6 7v10q0 4-4 4Z"
          fill={CAB_YELLOW}
          stroke={ASPHALT}
          strokeWidth="2.4"
          strokeLinejoin="round"
        />

        <path d="M36 33l6.6-8.8q1.2-1.7 3.2-1.7H58V33Z" fill={GLASS} />
        <path d="M62 22.5h12.5q2 0 3.1 1.7L84.5 33H62Z" fill={GLASS} />
        <path d="M66 24.2h4l-5 7h-2.6Z" fill="#fff" opacity=".2" />
        <path d="M47 24.2h4l-5 7h-2.6Z" fill="#fff" opacity=".14" />

        <rect x="12" y="42" width="96" height="6" fill={`url(#${patternId})`} />

        <path d="M60 33v21" stroke={ASPHALT} strokeWidth="1.5" opacity=".5" />
        <rect x="48" y="37" width="6" height="1.6" rx=".8" fill={ASPHALT} />
        <rect x="66" y="37" width="6" height="1.6" rx=".8" fill={ASPHALT} />

        <rect x="103.5" y="37.5" width="5" height="3.8" rx="1.5" fill="#FFF3B0" stroke={ASPHALT} strokeWidth="1.2" />
        <rect x="11" y="37.5" width="3.4" height="3.8" rx="1" fill="#E5484D" />

        <rect x="99" y="52" width="13" height="3.6" rx="1.8" fill="#2A2E36" />
        <rect x="8" y="52" width="11" height="3.6" rx="1.8" fill="#2A2E36" />
      </g>

      <g className="taxi-wheel">
        <circle cx="32" cy="56" r="8.5" fill={ASPHALT} />
        <circle cx="32" cy="56" r="4" fill="#D5DCE2" />
        <circle cx="32" cy="56" r="1.5" fill={ASPHALT} />
      </g>
      <g className="taxi-wheel">
        <circle cx="88" cy="56" r="8.5" fill={ASPHALT} />
        <circle cx="88" cy="56" r="4" fill="#D5DCE2" />
        <circle cx="88" cy="56" r="1.5" fill={ASPHALT} />
      </g>
    </svg>
  )
}

export function CheckerStrip({ className = '' }: { className?: string }) {
  return <span className={`checker-strip ${className}`} aria-hidden="true" />
}
