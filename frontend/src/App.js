    import React, { useState } from "react";
    import Login from "./login"; // Assuming Login.js is in the same directory
    import Encode from "./components/Encode"; // Corrected path to Encode.js

    function App() {
      const [loggedInUser, setLoggedInUser] = useState(null);

      // If not logged in, show the Login component
      if (!loggedInUser) {
        // Pass a function to set the logged-in user upon successful login
        return <Login onLoginSuccess={setLoggedInUser} />;
      }

      // If logged in, show the Encode component
      return (
        <div style={{ fontFamily: "'Inter', sans-serif", backgroundColor: "#f8f9fa", minHeight: "100vh", padding: "2rem 0" }}>
          <h1 style={{ textAlign: "center", color: "#343a40", marginBottom: "2rem" }}>
            Welcome, {loggedInUser}!
          </h1>
          <Encode />
        </div>
      );
    }

    export default App;
    