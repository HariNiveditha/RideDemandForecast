css_path = r"C:\Users\mitta\RideDemandForecast\frontend\src\App.css"

fleet_css = """

/* === Fleet Management Operational Workspace Styles === */
.secondary-button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #fff;
    color: var(--ink);
    padding: 9px 14px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: background .15s ease, border-color .15s ease, transform .15s ease;
}

.secondary-button:hover {
    background: #f8fafc;
    border-color: #cbd5e1;
}

.secondary-button:active {
    transform: translateY(1px);
}

.secondary-button:disabled,
.primary-button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
}

.fleet-kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

@media (max-width: 1280px) {
    .fleet-kpi-grid {
        grid-template-columns: repeat(3, 1fr);
    }
}

@media (max-width: 768px) {
    .fleet-kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.fleet-kpi-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 11px;
    box-shadow: var(--shadow);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.fleet-kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.fleet-kpi-icon {
    color: var(--teal);
    display: flex;
    align-items: center;
}

.fleet-kpi-card strong {
    display: block;
    font-size: 24px;
    color: var(--navy);
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 10px 0 2px;
}

.fleet-kpi-card small {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.3;
}

.fleet-status-bar {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: var(--shadow);
    padding: 10px 16px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 12px;
}

.fleet-status-group {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}

.fleet-control-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 11px;
    box-shadow: var(--shadow);
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}

.fleet-control-inputs {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}

.fleet-control-actions {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.fleet-select-wrapper {
    display: inline-flex;
    flex-direction: column;
    gap: 4px;
}

.fleet-select-wrapper label {
    font-size: 11px;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
}

.fleet-select-wrapper select {
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid var(--line);
    background: #fff;
    color: var(--navy);
    font-size: 13px;
    font-weight: 600;
    outline: none;
    cursor: pointer;
    transition: border-color .15s ease, box-shadow .15s ease;
}

.fleet-select-wrapper select:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px rgba(42, 166, 164, 0.15);
}

.fleet-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 11px;
    box-shadow: var(--shadow);
    padding: 20px;
    margin-bottom: 20px;
}

.fleet-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 14px;
    flex-wrap: wrap;
}

.fleet-card-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--navy);
    margin: 0;
}

.fleet-card-subtitle {
    font-size: 12px;
    color: var(--muted);
}

.fleet-disclosure-box {
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 12px;
    line-height: 1.5;
}

.fleet-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

.fleet-pill-success {
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}

.fleet-pill-info {
    background: #eff6ff;
    color: #1e40af;
    border: 1px solid #bfdbfe;
}

.fleet-pill-warning {
    background: #fffbeb;
    color: #92400e;
    border: 1px solid #fde68a;
}

.fleet-pill-danger {
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.fleet-recs-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.fleet-recs-table th {
    background: #f8fafc;
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--line);
}

.fleet-recs-table td {
    padding: 11px 14px;
    border-bottom: 1px solid #f1f5f9;
    color: var(--ink);
    vertical-align: middle;
}

.fleet-recs-table tr:hover td {
    background: #f8fafc;
}

.fleet-cab-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: #fef3c7;
    color: #92400e;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 12px;
}

.fleet-notice-banner {
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.fleet-notice-success {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}

.fleet-notice-error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.fleet-stats-strip {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--line);
}

.fleet-stats-strip strong {
    color: var(--navy);
}
"""

with open(css_path, "r", encoding="utf-8") as f:
    current = f.read()

if "Fleet Management Operational Workspace Styles" not in current:
    with open(css_path, "a", encoding="utf-8") as f:
        f.write(fleet_css)
    print("APPENDED FLEET CSS SUCCESSFULLY")
else:
    print("FLEET CSS ALREADY PRESENT")
