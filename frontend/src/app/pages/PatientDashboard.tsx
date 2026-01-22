import { useState } from "react";

const PatientDashboard = () => {
  const [showPopup, setShowPopup] = useState(false);
  const [result, setResult] = useState("");
  const [form, setForm] = useState({
    diagnosis: "Obesity",
    heart_rate: "",
    respiratory_rate: "",
    oxygen_saturation: ""
  });

  const recommendTreatment = async () => {
    const res = await fetch("http://localhost:8080/treatment/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        age: 30,
        gender: "Female",
        diagnosis: form.diagnosis,
        heart_rate: Number(form.heart_rate),
        respiratory_rate: Number(form.respiratory_rate),
        oxygen_saturation: Number(form.oxygen_saturation)
      })
    });

    const data = await res.text();
    setResult(data);
  };

  return (
    <div>
      <h2>Patient Dashboard</h2>
      <p><b>Name:</b> irfan</p>
      <p><b>Age:</b> 30</p>
      <p><b>Gender:</b> Female</p>

      <button onClick={() => setShowPopup(true)}>Treatment Recommendation</button>

      {showPopup && (
        <div style={{ border: "1px solid black", padding: 20, marginTop: 20 }}>
          <h3>Recommend Treatment</h3>

          <select onChange={e => setForm({ ...form, diagnosis: e.target.value })}>
            <option>Type 2 Diabetes</option>
            <option>Asthma</option>
            <option>Hypertension</option>
            <option>Obesity</option>
          </select>

          <input placeholder="Heart Rate" onChange={e => setForm({ ...form, heart_rate: e.target.value })} />
          <input placeholder="Respiratory Rate" onChange={e => setForm({ ...form, respiratory_rate: e.target.value })} />
          <input placeholder="Oxygen Saturation" onChange={e => setForm({ ...form, oxygen_saturation: e.target.value })} />

          <button onClick={recommendTreatment}>Recommend</button>

          {result && <p><b>Recommended Treatment:</b> {result}</p>}

          <button onClick={() => setShowPopup(false)}>Close</button>
        </div>
      )}
    </div>
  );
};

export default PatientDashboard;
