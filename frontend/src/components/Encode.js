import React, { useState } from 'react';

// Reusable component for styled sections
const Section = ({ title, children }) => (
  <div style={{ padding: "1.5rem", border: "1px solid #ddd", borderRadius: "8px", boxShadow: "0 2px 5px rgba(0,0,0,0.05)", backgroundColor: "#fff" }}>
    <h2 style={{ marginBottom: "1.5rem", textAlign: "center", color: "#333", borderBottom: "2px solid #007bff", paddingBottom: "0.5rem" }}>{title}</h2>
    {children}
  </div>
);

function Encode() {
  // Encoding States
  const [inputType, setInputType] = useState('text'); // 'text' or 'image'
  const [text, setText] = useState('');
  const [secretImage, setSecretImage] = useState(null);
  const [coverImage, setCoverImage] = useState(null);
  const [encodePassword, setEncodePassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [encodeProgress, setEncodeProgress] = useState(0);
  const [response, setResponse] = useState('');
  const [encodedImage, setEncodedImage] = useState(null);

  // Decoding States
  const [decodeType, setDecodeType] = useState('text'); // 'text' or 'image'
  const [decodeImageFile, setDecodeImageFile] = useState(null);
  const [decodePassword, setDecodePassword] = useState('');
  const [decodingLoading, setDecodingLoading] = useState(false);
  const [decodeResponse, setDecodeResponse] = useState('');
  const [decodedMessage, setDecodedMessage] = useState('');
  const [decodedImageUrl, setDecodedImageUrl] = useState(null);

  // --- API HANDLERS ---
  const pollJobStatus = (jobId) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:5000/status/${jobId}`);
        const data = await res.json();
        if (data.status === 'processing') {
          setEncodeProgress(data.progress);
        } else if (data.status === 'complete') {
          clearInterval(interval);
          setLoading(false);
          setResponse('Encoding successful!');
          setEncodeProgress(100);
          setEncodedImage(data.encoded_image_url);
        } else if (data.status === 'error') {
          clearInterval(interval);
          setLoading(false);
          setResponse(`Error: ${data.message}`);
        }
      } catch (err) {
        clearInterval(interval);
        setLoading(false);
        setResponse('Error: Could not poll job status.');
      }
    }, 2000);
  };

  const handleEncode = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse('Starting encoding process...');
    setEncodedImage(null);
    setEncodeProgress(0);

    const formData = new FormData();
    formData.append('inputType', inputType);
    formData.append('coverImage', coverImage);
    if (encodePassword) formData.append('password', encodePassword);

    if (inputType === 'text') {
      formData.append('text', text);
    } else {
      formData.append('secretImage', secretImage);
    }

    try {
      const res = await fetch('http://localhost:5000/encode', { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok && data.job_id) {
        setResponse('Processing... Please wait.');
        pollJobStatus(data.job_id);
      } else {
        setResponse(`Error: ${data.message || 'Failed to start encoding.'}`);
        setLoading(false);
      }
    } catch (err) {
      setResponse('Error: Could not connect to the server.');
      setLoading(false);
    }
  };

  const handleDecode = async (e) => {
    e.preventDefault();
    setDecodingLoading(true);
    setDecodeResponse('Decoding in progress...');
    setDecodedMessage('');
    setDecodedImageUrl(null);

    const formData = new FormData();
    formData.append('image', decodeImageFile);
    formData.append('decodeType', decodeType); // Specify what to extract
    if (decodePassword) formData.append('password', decodePassword);

    try {
      const res = await fetch('http://localhost:5000/decode', { method: 'POST', body: formData });
      const data = await res.json();
      setDecodeResponse(data.message);
      if (res.ok) {
        if (data.decoded_type === 'text') setDecodedMessage(data.decoded_content);
        if (data.decoded_type === 'image') setDecodedImageUrl(data.decoded_content_url);
      }
    } catch (err) {
      setDecodeResponse('Error: Could not connect to the server.');
    } finally {
      setDecodingLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: "800px", margin: "auto", padding: "1rem", display: "grid", gridTemplateColumns: "1fr", gap: "2rem" }}>
      {/* --- ENCODING SECTION --- */}
      <Section title="Encode Data">
        <form onSubmit={handleEncode} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <label>1. Select Input Type:</label>
          <select value={inputType} onChange={(e) => setInputType(e.target.value)} style={{ width: "100%", padding: "0.5rem" }}>
            <option value="text">Text in Image</option>
            <option value="image">Image in Image</option>
          </select>

          {inputType === 'text' ? (
            <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Enter your secret message..." rows={4} required style={{ width: "100%", padding: "0.5rem" }} />
          ) : (
            <label>Upload Secret Image: <input type="file" accept="image/*" onChange={(e) => setSecretImage(e.target.files[0])} required style={{ width: "100%" }} /></label>
          )}

          <label>Upload Cover Image: <input type="file" accept="image/*" onChange={(e) => setCoverImage(e.target.files[0])} required style={{ width: "100%" }} /></label>
          <label>Encryption Password (Optional): <input type="password" value={encodePassword} onChange={(e) => setEncodePassword(e.target.value)} style={{ width: "100%", padding: "0.5rem" }} /></label>

          <button type="submit" disabled={loading} style={{ padding: "0.8rem", cursor: "pointer" }}>{loading ? `Processing... ${encodeProgress}%` : "Encode"}</button>
          {response && <p style={{ backgroundColor: "#f0f0f0", padding: "0.5rem", borderRadius: "4px" }}>{response}</p>}
          {encodedImage && (
            <div style={{ textAlign: "center", marginTop: "1rem" }}>
              <h4>Encoded Image Ready:</h4>
              <img src={encodedImage} alt="Encoded" style={{ maxWidth: "100%", borderRadius: "8px" }} />
              <a href={encodedImage} download style={{ display: "block", marginTop: "0.5rem" }}>Download Image</a>
            </div>
          )}
        </form>
      </Section>

      {/* --- DECODING SECTION --- */}
      <Section title="Decode Data">
        <form onSubmit={handleDecode} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <label>1. Select What to Extract:</label>
          <select value={decodeType} onChange={(e) => setDecodeType(e.target.value)} style={{ width: "100%", padding: "0.5rem" }}>
            <option value="text">Extract Text</option>
            <option value="image">Extract Image</option>
          </select>

          <label>Upload Stego Image: <input type="file" accept="image/*" onChange={(e) => setDecodeImageFile(e.target.files[0])} required style={{ width: "100%" }} /></label>
          <label>Decryption Password (if used): <input type="password" value={decodePassword} onChange={(e) => setDecodePassword(e.target.value)} style={{ width: "100%", padding: "0.5rem" }} /></label>

          <button type="submit" disabled={decodingLoading} style={{ padding: "0.8rem", cursor: "pointer" }}>{decodingLoading ? "Decoding..." : "Decode"}</button>
          {decodeResponse && <p style={{ backgroundColor: "#f0f0f0", padding: "0.5rem", borderRadius: "4px" }}>{decodeResponse}</p>}
          {decodedMessage && (
            <div style={{ marginTop: "1rem" }}>
              <h4>Decoded Text:</h4>
              <p style={{ wordBreak: "break-all" }}>{decodedMessage}</p>
            </div>
          )}
          {decodedImageUrl && (
            <div style={{ textAlign: "center", marginTop: "1rem" }}>
              <h4>Decoded Image:</h4>
              <img src={decodedImageUrl} alt="Decoded" style={{ maxWidth: "100%", borderRadius: "8px" }} />
              <a href={decodedImageUrl} download style={{ display: "block", marginTop: "0.5rem" }}>Download Decoded Image</a>
            </div>
          )}
        </form>
      </Section>
    </div>
  );
}

export default Encode;
