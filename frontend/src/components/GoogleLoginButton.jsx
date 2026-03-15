import { useEffect, useRef } from 'react';

const GOOGLE_CLIENT_ID =
  '729857577095-k5p6nbu7s8dnj8kh8sd6pvfehuiji4fp.apps.googleusercontent.com';
const API_BASE = 'http://127.0.0.1:8000';

export default function GoogleLoginButton({
  onLoginSuccess,
  loginType = 'dashboard',
  buttonText = 'signin_with',
  width = 280,
}) {
  const buttonRef = useRef(null);

  useEffect(() => {
    if (!window.google || !buttonRef.current) return;

    buttonRef.current.innerHTML = '';

    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: async (response) => {
        try {
          const res = await fetch(`${API_BASE}/auth/google`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              credential: response.credential,
              login_type: loginType,
            }),
          });

          const data = await res.json();

          if (!res.ok) {
            throw new Error(data.detail || 'Google login failed');
          }

          const userPayload = {
            ...data.user,
            login_type: data.login_type || loginType,
          };

          localStorage.setItem('trustlens_user', JSON.stringify(userPayload));
          onLoginSuccess?.(userPayload);
        } catch (error) {
          console.error('Google login failed:', error);
          alert(error.message || 'Google login failed');
        }
      },
    });

    window.google.accounts.id.renderButton(buttonRef.current, {
      theme: 'outline',
      size: 'large',
      text: buttonText,
      shape: 'pill',
      width,
    });
  }, [onLoginSuccess, loginType, buttonText, width]);

  return <div ref={buttonRef}></div>;
}