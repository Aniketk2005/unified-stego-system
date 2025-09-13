    import React, { useState } from "react";

    function Login({ onLoginSuccess }) {
      const [username, setUsername] = useState("");
      const [password, setPassword] = useState("");
      const [error, setError] = useState("");
      const [loading, setLoading] = useState(false);

      const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");

        try {
          const res = await fetch("http://localhost:5000/login", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ username, password }),
          });

          const data = await res.json();
          if (res.ok) { // Check for 2xx status codes
            onLoginSuccess(username); // Call the prop function to update parent state
          } else {
            setError(data.message || "Login failed. Please try again.");
          }
        } catch (err) {
          console.error("Login error:", err);
          setError("Network error. Could not connect to the server.");
        } finally {
          setLoading(false);
        }
      };

      return (
        <div style={{ maxWidth: "400px", margin: "auto", padding: "2rem", border: "1px solid #ccc", borderRadius: "8px", boxShadow: "0 2px 10px rgba(0,0,0,0.1)" }}>
          <h2 style={{ textAlign: "center", marginBottom: "1.5rem" }}>Login</h2>
          <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <label>
              Username:
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
              />
            </label>
            <label>
              Password:
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
              />
            </label>
            {error && <p style={{ color: "red", textAlign: "center" }}>{error}</p>}
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: "0.8rem 1.5rem",
                borderRadius: "5px",
                border: "none",
                backgroundColor: "#007bff",
                color: "white",
                fontSize: "1rem",
                cursor: "pointer",
                transition: "background-color 0.3s ease",
              }}
            >
              {loading ? "Logging in..." : "Login"}
            </button>
          </form>
        </div>
      );
    }

    export default Login;
    