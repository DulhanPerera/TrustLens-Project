import { useEffect, useRef } from 'react';

const GOOGLE_CLIENT_ID = '729857577095-k5p6nbu7s8dnj8kh8sd6pvfehuiji4fp.apps.googleusercontent.com';
const API_BASE = 'http://127.0.0.1:8000';

export default function GoogleLoginButton({ onLoginSuccess }) {
  const buttonRef = useRef(null);

  useEffect(() => {
    if (!window.google || !buttonRef.current) return;

    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: async (response) => {
        try {
          const res = await fetch(`${API_BASE}/auth/google`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ credential: response.credential }),
          });

          const data = await res.json();

          if (!res.ok) {
            throw new Error(data.detail || 'Google login failed');
          }

          localStorage.setItem('trustlens_user', JSON.stringify(data.user));
          onLoginSuccess?.(data.user);
        } catch (error) {
          console.error('Google login failed:', error);
          alert('Google login failed');
        }
      },
    });

    window.google.accounts.id.renderButton(buttonRef.current, {
      theme: 'outline',
      size: 'large',
      text: 'signin_with',
      shape: 'pill',
      width: 280,
    });
  }, [onLoginSuccess]);

  return <div ref={buttonRef}></div>;
}