import { useState } from "react";
import BottomNav from "@/components/BottomNav";
import HomeScreen from "@/components/HomeScreen";
import ListenScreen from "@/components/ListenScreen";
import HeatmapScreen from "@/components/HeatmapScreen";
import AdvisorScreen from "@/components/AdvisorScreen";
import ProfileScreen from "@/components/ProfileScreen";
import SystemPanel from "@/components/SystemPanel";

type Tab = "home" | "listen" | "map" | "advisor" | "profile";

const Index = () => {
  const [activeTab, setActiveTab] = useState<Tab>("home");
  const [showSystem, setShowSystem] = useState(false);

  if (showSystem) {
    return <SystemPanel onBack={() => setShowSystem(false)} />;
  }

  return (
    <div className="mx-auto min-h-screen max-w-md bg-background px-4">
      {activeTab === "home" && (
        <HomeScreen onNavigate={setActiveTab} onOpenSystem={() => setShowSystem(true)} />
      )}
      {activeTab === "listen" && <ListenScreen />}
      {activeTab === "map" && <HeatmapScreen />}
      {activeTab === "advisor" && <AdvisorScreen />}
      {activeTab === "profile" && <ProfileScreen />}
      <BottomNav active={activeTab} onNavigate={setActiveTab} />
    </div>
  );
};

export default Index;
