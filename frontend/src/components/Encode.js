import React, { useState, useEffect } from "react";

function Encode() {
  // All your state variables remain the same
  const [inputType, setInputType] = useState("text");
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [coverImageFile, setCoverImageFile] = useState(null);
  const [outputMedia, setOutputMedia] = useState("image");
  const [medium, setMedium] = useState("WhatsApp");
  const [confidentiality, setConfidentiality] = useState("casual");
  const [allowAI, setAllowAI] = useState(false); // Defaulting AI to off for simplicity
  const [manualEncoding, setManualEncoding] = useState("lsb_image");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [encodedImage, setEncodedImage] = useState(null);
  const [encodeProgress, setEncodeProgress] = useState(0);
  const [encodePassword, setEncodePassword] = useState("");
  const [decodeImageFile, setDecodeImageFile] = useState(null);
  const [decodedMessage, setDecodedMessage] = useState("");
  const [decodedImageUrl, setDecodedImageUrl] = useState(null);
  const [decodingLoading, setDecodingLoading] = useState(false);
  const [decodeResponse, setDecodeResponse] = useState("");
  const [decodePassword, setDecodePassword] = useState("");

  // This polling interval will be used to check the status
  let pollingInterval;

  const pollJobStatus = async (jobId) => {
    pollingInterval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:5000/status/${jobId}`);
        const data = await res.json();

        if (data.status === 'processing') {
          setEncodeProgress(data.progress);
          setResponse("Encoding in progress...");
        } else if (data.status === 'complete') {
          clearInterval(pollingInterval);
          setLoading(false);
          setResponse("Encoding successful!");
          setEncodeProgress(100);
          // THIS IS THE KEY: Set the image URL to display it
          setEncodedImage(data.encoded_image_url);
        } else if (data.status === 'error') {
          clearInterval(pollingInterval);
          setLoading(false);
          setResponse(`Error: ${data.message}`);
          setEncodeProgress(0);
        }
      } catch (err) {
        clearInterval(pollingInterval);
        setLoading(false);
        setResponse("Error: Could not poll job status.");
        console.error("Polling error:", err);
      }
    }, 2000); // Check every 2 seconds
  };

  const handleEncode = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse("Uploading files...");
    setEncodedImage(null);
    setDecodedMessage("");
    setEncodeProgress(0);

    let formData = new FormData();
    formData.append("text", text);
    formData.append("coverImage", coverImageFile);
    if (encodePassword) {
      formData.append("password", encodePassword);
    }

    try {
      // Step 1: Send the request to /encode
      const res = await fetch("http://localhost:5000/encode", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok && data.job_id) {
        // Step 2: If we get a job_id, start polling for the status
        setResponse("Processing... Please wait.");
        pollJobStatus(data.job_id);
      } else {
        setResponse(`Error: ${data.message || 'Failed to start encoding job.'}`);
        setLoading(false);
      }
    } catch (err) {
      console.error("Error during encode submission:", err);
      setResponse("Error connecting to server to start encoding.");
      setLoading(false);
    }
  };

  const handleDecode = async (e) => {
    e.preventDefault();
    setDecodingLoading(true);
    setDecodedMessage("");
    setDecodedImageUrl(null);
    setDecodeResponse("");

    if (!decodeImageFile) {
      setDecodeResponse("Please select an image to decode.");
      setDecodingLoading(false);
      return;
    }

    const formData = new FormData();
    formData.append("image", decodeImageFile);
    if (decodePassword) formData.append("password", decodePassword);

    try {
      const res = await fetch("http://localhost:5000/decode", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setDecodedMessage(data.decoded_content);
        setDecodeResponse("Decoding successful!");
      } else {
        setDecodeResponse(`Error: ${data.message || 'Decoding failed.'}`);
      }
    } catch (err) {
      console.error("Error during decoding:", err);
      setDecodeResponse("Error connecting to server for decoding.");
    } finally {
      setDecodingLoading(false);
    }
  };

  // Your JSX remains largely the same, just remove the simulated progress effects
  // The rest of your component (return statement with JSX) goes here...
  return (
    <div style={{ maxWidth: "800px", margin: "auto", padding: "1rem", display: "grid", gridTemplateColumns: "1fr", gap: "2rem" }}>
      {/* Encoding Section */}
      <div style={{ padding: "1.5rem", border: "1px solid #ddd", borderRadius: "8px", boxShadow: "0 2px 5px rgba(0,0,0,0.05)", backgroundColor: "#fff" }}>
        <h2 style={{ marginBottom: "1rem", textAlign: "center", color: "#333" }}>Encode Your Message</h2>
        <form onSubmit={handleEncode} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {/* We'll simplify the UI to focus on the working parts */}
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter your secret message"
            rows={4}
            required
            style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
          />
          <label>
            Upload Cover Image:
            <input
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              onChange={(e) => setCoverImageFile(e.target.files[0])}
              required
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
            />
          </label>
          <label>
            Encryption Password (Optional):
            <input
              type="password"
              value={encodePassword}
              onChange={(e) => setEncodePassword(e.target.value)}
              placeholder="Leave blank for no encryption"
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
            />
          </label>
          <button
            type="submit"
            style={{
              marginTop: "1rem", padding: "0.8rem 1.5rem", borderRadius: "5px", border: "none",
              backgroundColor: "#007bff", color: "white", fontSize: "1rem", cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem"
            }}
            disabled={loading}
          >
            {loading ? "Processing..." : "Encode"}
          </button>
          {loading && (
            <div>
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${encodeProgress}%` }}></div>
              </div>
              <div className="progress-text">{encodeProgress}%</div>
            </div>
          )}
        </form>

        {response && <p style={{ marginTop: "1rem", padding: "0.5rem", borderRadius: "4px", backgroundColor: "#e9ecef" }}>{response}</p>}

        {encodedImage && (
          <div style={{ marginTop: "1rem", textAlign: "center" }}>
            <h3>Encoded Image Ready:</h3>
            <img src={encodedImage} alt="Encoded" style={{ maxWidth: "100%", border: "1px solid #ccc", borderRadius: "8px" }}/>
            <br />
            <a href={encodedImage} download style={{
              display: "inline-block", marginTop: "1rem", padding: "0.5rem 1rem", borderRadius: "5px",
              border: "1px solid #007bff", backgroundColor: "transparent", color: "#007bff", textDecoration: "none"
            }}>
              Download Encoded Image
            </a>
          </div>
        )}
      </div>

      {/* Decoding Section */}
      <div style={{ padding: "1.5rem", border: "1px solid #ddd", borderRadius: "8px", boxShadow: "0 2px 5px rgba(0,0,0,0.05)", backgroundColor: "#fff" }}>
        <h2 style={{ marginBottom: "1rem", textAlign: "center", color: "#333" }}>Decode Your Message</h2>
        <form onSubmit={handleDecode} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <label>
            Upload Encoded Image:
            <input
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              onChange={(e) => setDecodeImageFile(e.target.files[0])}
              required
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
            />
          </label>
          <label>
            Decryption Password (If encrypted):
            <input
              type="password"
              value={decodePassword}
              onChange={(e) => setDecodePassword(e.target.value)}
              placeholder="Enter password if encrypted"
              style={{ width: "100%", padding: "0.5rem", borderRadius: "4px", border: "1px solid #ccc" }}
            />
          </label>
          <button
            type="submit"
            style={{
              marginTop: "1rem", padding: "0.8rem 1.5rem", borderRadius: "5px", border: "none",
              backgroundColor: "#28a745", color: "white", fontSize: "1rem", cursor: "pointer"
            }}
            disabled={decodingLoading}
          >
            {decodingLoading ? "Decoding..." : "Decode Image"}
          </button>
        </form>

        {decodeResponse && <p style={{ marginTop: "1rem", padding: "0.5rem", borderRadius: "4px", backgroundColor: "#e9ecef" }}>{decodeResponse}</p>}

        {decodedMessage && (
          <div style={{ marginTop: "1rem", padding: "1rem", border: "1px solid #007bff", borderRadius: "8px", backgroundColor: "#e6f7ff" }}>
            <h3>Decoded Message:</h3>
            <p style={{ fontWeight: "bold", wordBreak: "break-all" }}>{decodedMessage}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Encode;