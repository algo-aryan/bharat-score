# Bank of India Rural Credit Assessment System

A responsive web application for assessing rural applicant creditworthiness using alternative data (DBT schemes, bill payments, AEPS/UPI transactions, cash flow, geo-stability) powered by AI & Machine Learning. Includes an interactive chatbot assistant for in-page guidance.

## Features

### Applicant Data Capture
- Collects name, Aadhaar, mobile, location, residence years, requested loan amount.
- Automatically fetches financial data via simulated APIs: DBT payments, bill payment ratio, AEPS/UPI transactions, cash flow ratio, geo-stability.

### Loading Progress Overlay
- Multi-step simulation with progress indicator for each data source and ML processing.

### Automated Score Generation
Weighted scoring to compute:
- **Credit Score** (300–900 scale)
- **Risk Score** (default probability)
- **Loan Recommendation** (Approve, Review, Reject)

### Results Dashboard
- Displays fetched data cards, score cards, component progress bars, key risk factors list.
- Doughnut chart for score composition.
- Bar chart for component impact analysis.

### Chatbot Assistant
- Floating chat icon at bottom-right.
- Dialog popup with history, user & bot message styling.
- Basic script simulating query analysis and response suggestions.

### Responsive Design
- Adapts to desktop and mobile screen widths.

## Installation & Setup
Clone repository:
```bash
git clone https://github.com/your-org/rural-credit-assessment.git
```

Open in browser:
- Open `index.html` in any modern browser.
- No server required.

## Usage

### New Assessment
1. Fill out applicant form.
2. Click **Assess Credit Score**.

### Watch Progress
- Loading overlay visualizes data fetching & ML steps.

### View Results
- Dashboard shows fetched metrics, computed scores, charts, risk factors, and recommendation.

### Use Chatbot
- Click the bottom-right chat icon.
- Enter queries (e.g., “Explain risk factors”).
- Bot responds with static guidance messages.

### Start Over
- Click **New Assessment** on results screen to reset.

## Customization
- **Data Integration**: Replace simulation functions with real API endpoints to fetch actual financial data.
- **Chatbot Logic**: Integrate with Dialogflow or custom NLP backend by updating message handlers.
- **Styling**: Modify CSS variables under `:root` to adjust color themes.

## License
See the accompanying LICENSE file for details.
