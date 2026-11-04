# KNN + SVM Virtual Lab

An interactive educational virtual laboratory for comparing **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**. Built with React and Vite.

## Features
- Interactive 2D scatter plot showing training samples, query point, and nearest-neighbor connections
- Adjustable odd K values (1, 3, 5, 7, 9)
- Euclidean and Manhattan distance metrics
- Live nearest-neighbor ranking, distances, and majority-vote prediction
- Two sample datasets: flower classification and student performance
- Guided experiments, concept explanations, and a knowledge-check assessment
- Linear, polynomial and RBF-kernel SVM simulation with adjustable C and gamma parameters
- Visual decision regions, support vectors and approximate margin bands
- Side-by-side KNN and SVM prediction and performance comparison
- Theory section covering KNN, SVM, kernels, margins and model selection
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
2. Explain SVM hyperplanes, support vectors and margins.
3. Compare linear and kernelized SVM models with KNN.
2. Understand how K affects predictions.
3. Compare Euclidean and Manhattan distances.
4. Identify nearest neighbors and apply majority voting.

The included datasets are illustrative teaching data, not a validated benchmark dataset.
