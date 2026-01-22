import { useState } from "react";
import { useNavigate } from "react-router-dom";

const Signup = () => {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    username: "",
    password: "",
    name: "",
    age: "",
    gender: ""
  });

  const handleSignup = async () => {
    const res = await fetch("http://localhost:8080/user/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...form,
        age: Number(form.age)
      })
    });

    if (res.ok) {
      alert("Signup successful");
      navigate("/");
    } else {
      alert("Signup failed");
    }
  };

  return (
    <div>
      <h2>Signup</h2>
      <input placeholder="Username" onChange={e => setForm({ ...form, username: e.target.value })} />
      <input type="password" placeholder="Password" onChange={e => setForm({ ...form, password: e.target.value })} />
      <input placeholder="Name" onChange={e => setForm({ ...form, name: e.target.value })} />
      <input placeholder="Age" onChange={e => setForm({ ...form, age: e.target.value })} />
      <input placeholder="Gender" onChange={e => setForm({ ...form, gender: e.target.value })} />
      <button onClick={handleSignup}>Signup</button>
    </div>
  );
};

export default Signup;
