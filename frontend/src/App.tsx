import { Component, type ReactNode } from 'react';
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom';
import AppShell from './components/AppShell';
import { WardProvider } from './context/WardContext';
import Calibrate from './pages/Calibrate';
import Dashboard from './pages/Dashboard';
import PatientDetail from './pages/PatientDetail';
import System from './pages/System';

/** A render error in one view must never blank the whole monitor. */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="panel p-6 max-w-[70ch] flex flex-col gap-3">
        <p className="t-head">This view hit an error.</p>
        <p className="t-label text-ink2 break-words">{this.state.error.message}</p>
        <button type="button" className="btn w-fit" onClick={() => this.setState({ error: null })}>Try again</button>
      </div>
    );
  }
}

function NotFound() {
  return (
    <div className="py-16 flex flex-col gap-4">
      <p className="font-cond font-semibold text-[48px] leading-none">Page not found.</p>
      <Link to="/" className="btn w-fit">Back to the ward</Link>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <WardProvider>
        <AppShell>
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/patients/:id" element={<PatientDetail />} />
              <Route path="/patients/:id/calibrate" element={<Calibrate />} />
              <Route path="/system" element={<System />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </ErrorBoundary>
        </AppShell>
      </WardProvider>
    </BrowserRouter>
  );
}
