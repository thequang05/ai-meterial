import { Routes, Route } from "react-router-dom";
import { Layout } from "@/components/Layout";
import { Landing } from "@/pages/Landing";
import { Campaigns } from "@/pages/Campaigns";
import { CandidateDetail } from "@/pages/CandidateDetail";
import { Pipeline } from "@/pages/Pipeline";
import { NLQuery } from "@/pages/NLQuery";
import { About } from "@/pages/About";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/campaigns" element={<Campaigns />} />
        <Route path="/campaigns/:id" element={<CandidateDetail />} />
        <Route path="/pipeline" element={<Pipeline />} />
        <Route path="/nl-query" element={<NLQuery />} />
        <Route path="/about" element={<About />} />
      </Route>
    </Routes>
  );
}
