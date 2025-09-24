import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders symbol control and forecast section', () => {
  render(<App />);
  expect(screen.getByText(/symbol/i)).toBeInTheDocument();
  expect(screen.getByText(/markov forecast/i)).toBeInTheDocument();
});
