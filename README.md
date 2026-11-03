# KNN Virtual Lab

An interactive educational virtual laboratory for understanding the **K-Nearest Neighbors (KNN)** machine learning algorithm. Built with React and Vite.

## Features
- Interactive 2D scatter plot showing training samples, query point, and nearest-neighbor connections
- Adjustable odd K values (1, 3, 5, 7, 9)
- Euclidean and Manhattan distance metrics
- Live nearest-neighbor ranking, distances, and majority-vote prediction
- Two sample datasets: flower classification and student performance
- Guided experiments, concept explanations, and a knowledge-check assessment
- Responsive interface

## Run locally
Requires Node.js 18 or later.

```bash
npm install
npm run dev
```

## Production build
```bash
npm run build
npm run preview
```

## Deploy
The included GitHub Actions workflow builds and deploys the project to GitHub Pages on pushes to `main`. In repository Settings → Pages, select **GitHub Actions** as the build and deployment source.

## Learning outcomes
1. Explain the KNN classification process.
2. Understand how K affects predictions.
3. Compare Euclidean and Manhattan distances.
4. Identify nearest neighbors and apply majority voting.

The included datasets are illustrative teaching data, not a validated benchmark dataset.
