import { 
  Search, Activity, TrendingUp, TrendingDown, 
  LayoutDashboard, LineChart, Settings, Star, AlertCircle, ChevronDown, Bell, Zap, Trash2, Sun, Moon
} from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip as RechartsTooltip, CartesianGrid } from 'recharts';
import axios from 'axios';

function App() {
  const [view, setView] = useState('dashboard'); 
  const [ticker, setTicker] = useState('');
  const [data, setData] = useState(null);
  const [screenerData, setScreenerData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartRange, setChartRange] = useState('1Y');

  // --- THEME STATE ---
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('ai_trade_theme');
    return saved ? JSON.parse(saved) : true;
  });

  useEffect(() => {
    localStorage.setItem('ai_trade_theme', JSON.stringify(darkMode));
  }, [darkMode]);

  // --- EXPANDED DATA ---
  const watchlist = [
    { symbol: "RELIANCE", price: "₹2,954.10", change: "+1.2%", up: true },
    { symbol: "HDFCBANK", price: "₹1,432.50", change: "+0.8%", up: true },
    { symbol: "TCS", price: "₹3,982.10", change: "-1.2%", up: false },
    { symbol: "INFY", price: "₹1,643.20", change: "+2.1%", up: true },
    { symbol: "SBIN", price: "₹782.40", change: "+0.4%", up: true },
    { symbol: "TATAMOTORS", price: "₹964.00", change: "-0.9%", up: false },
    { symbol: "ICICIBANK", price: "₹1,084.20", change: "+1.5%", up: true },
    { symbol: "ADANIENT", price: "₹3,120.00", change: "+3.2%", up: true },
  ];

  const topMovers = [
    { symbol: "NIFTY 50", val: "22,453", change: "+0.56%", up: true },
    { symbol: "BANKNIFTY", val: "48,123", change: "+0.71%", up: true },
    { symbol: "GOLD", val: "72,140", change: "+1.12%", up: true }, // Added to endless scroll
  ];

  const marqueeStyle = (
    <style>{`
      @keyframes marquee {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
      }
      .animate-marquee {
        display: flex;
        width: max-content;
        animation: marquee 40s linear infinite;
      }
      .animate-marquee:hover {
        animation-play-state: paused;
      }
    `}</style>
  );

  const theme = {
    bg: darkMode ? 'bg-[#0a0f1c]' : 'bg-slate-50',
    sidebar: darkMode ? 'bg-[#0f172a]' : 'bg-white',
    card: darkMode ? 'bg-[#0f172a]' : 'bg-white',
    border: darkMode ? 'border-slate-800' : 'border-slate-200',
    textMain: darkMode ? 'text-white' : 'text-slate-900',
    textMuted: darkMode ? 'text-slate-400' : 'text-slate-500',
    input: darkMode ? 'bg-slate-900' : 'bg-slate-100',
  };

  const fetchAnalysis = async (targetTicker = ticker) => {
    if (!targetTicker) return;
    setLoading(true); setError(null);
    try {
      const res = await axios.get(`http://localhost:8000/api/predict/${targetTicker}`);
      setData(res.data); setView('dashboard');
    } catch (err) { setError("Backend Unreachable."); }
    setLoading(false);
  };

  const fetchScreener = async () => {
    setLoading(true); setView('screener');
    try {
      const res = await axios.get(`http://localhost:8000/api/screener`);
      setScreenerData(res.data);
    } catch (err) { setError("Screener failed."); }
    setLoading(false);
  };

  const filteredTrendData = useMemo(() => {
    if (!data?.trend_data) return [];
    switch (chartRange) {
      case '10D': return data.trend_data.slice(-10);
      case '1M': return data.trend_data.slice(-21);
      default: return data.trend_data;
    }
  }, [data, chartRange]);

  const goHome = () => {
    if (view === 'dashboard' && data) {
      setData(null);
      setTicker('');
    } else {
      setView('dashboard');
    }
  };

  return (
    <div className={`flex h-screen ${theme.bg} ${theme.textMain} font-sans overflow-hidden transition-colors duration-300`}>
      {marqueeStyle}
      
      {/* SIDEBAR */}
      <aside className={`w-64 ${theme.sidebar} border-r ${theme.border} flex flex-col hidden md:flex shrink-0 transition-colors duration-300`}>
        
        <div onClick={goHome} className={`h-20 flex items-center px-6 border-b ${theme.border} cursor-pointer group hover:${darkMode ? 'bg-slate-800/30' : 'bg-slate-50'} transition-all`}>
          <Zap className="text-emerald-500 mr-3 group-hover:scale-110 transition-transform" size={22} fill="currentColor" />
          <h1 className={`text-2xl font-black tracking-tighter ${theme.textMain}`}>PULSE.AI</h1>
        </div>
        
        <nav className="p-4 flex-1 overflow-y-auto space-y-8">
          {/* Main Links */}
          <div className="space-y-1.5">
            <button onClick={() => setView('dashboard')} className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl font-semibold transition-all ${view === 'dashboard' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : `${theme.textMuted} hover:${theme.textMain}`}`}><LayoutDashboard size={18} /> <span>Dashboard</span></button>
            <button onClick={fetchScreener} className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl font-semibold transition-all ${view === 'screener' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : `${theme.textMuted} hover:${theme.textMain}`}`}><LineChart size={18} /> <span>AI Screener</span></button>
          </div>
          
          {/* Watchlist Moved Down */}
          <div>
            <div className={`pb-2 px-3 text-[10px] font-black ${theme.textMuted} uppercase tracking-widest flex items-center justify-between`}>
              <span>Watchlist</span> <Star size={12} />
            </div>
            <div className="space-y-1">
              {watchlist.map((s, i) => (
                <div key={i} onClick={() => { setTicker(s.symbol + ".NS"); fetchAnalysis(s.symbol + ".NS"); }} className={`flex justify-between items-center px-3 py-2 hover:${darkMode ? 'bg-slate-800/50' : 'bg-slate-100'} rounded-lg cursor-pointer transition-all`}>
                  <div>
                    <div className={`text-xs font-bold ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>{s.symbol}</div>
                    <div className="text-[10px] text-slate-500">{s.price}</div>
                  </div>
                  <div className={`text-[10px] font-bold ${s.up ? 'text-emerald-500' : 'text-rose-500'}`}>{s.change}</div>
                </div>
              ))}
            </div>
          </div>
        </nav>

        <div className={`p-4 border-t ${theme.border}`}>
          <button onClick={() => setView('settings')} className={`flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors w-full ${view === 'settings' ? 'bg-emerald-500/10 text-emerald-500' : `${theme.textMuted} hover:${theme.textMain}`}`}>
            <Settings size={18} /> <span>Settings</span>
          </button>
        </div>
      </aside>

      {/* MAIN VIEWPORT */}
      <div className="flex-1 flex flex-col overflow-hidden">
        
        {/* HEADER WITH GOLD */}
        <header className={`h-20 border-b ${theme.border} ${theme.sidebar} backdrop-blur-md flex items-center overflow-hidden shrink-0 transition-colors duration-300`}>
           <div className="flex-1 overflow-hidden relative">
              <div className="animate-marquee flex items-center">
                 {[...topMovers, ...topMovers, ...topMovers].map((m, i) => (
                   <div key={i} className="flex items-center space-x-4 mx-12 whitespace-nowrap">
                     <span className={`${theme.textMuted} text-[10px] font-black uppercase tracking-widest`}>{m.symbol}</span>
                     <span className={`${theme.textMain} text-xl font-black tabular-nums`}>{m.val}</span>
                     <span className={`text-xs font-bold px-2 py-0.5 rounded-md ${m.up ? 'bg-emerald-500/10 text-emerald-500' : 'bg-rose-500/10 text-rose-500'}`}>{m.up ? '▲' : '▼'} {m.change}</span>
                   </div>
                 ))}
              </div>
           </div>
           
           <div className={`flex items-center space-x-4 px-6 border-l ${theme.border} h-full`}>
             <div className="relative">
                <input className={`${theme.input} border ${theme.border} rounded-xl px-4 py-2 pl-9 text-sm ${theme.textMain} outline-none focus:border-emerald-500 w-44`} value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} onKeyDown={(e) => e.key === 'Enter' && fetchAnalysis()} placeholder="Ticker..." />
                <Search className="absolute left-3 top-2.5 text-slate-500" size={14} />
             </div>
             <button onClick={() => fetchAnalysis()} className="bg-emerald-500 hover:bg-emerald-400 text-white px-5 py-2 rounded-xl text-xs font-black transition-all">ANALYZE</button>
           </div>
        </header>

        <main className="flex-1 overflow-y-auto p-8 scrollbar-hide">
          {error && <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-xl mb-6 flex items-center space-x-3 font-bold"><AlertCircle size={20} /> <span>{error}</span></div>}

          {loading ? (
            <div className="h-full flex flex-col items-center justify-center space-y-4">
               <div className={`w-12 h-12 border-[3px] ${theme.border} border-t-emerald-500 rounded-full animate-spin`}></div>
               <p className={`${theme.textMuted} font-black tracking-widest text-[10px] uppercase animate-pulse`}>Computing Inference...</p>
            </div>
          ) : view === 'dashboard' ? (
            data ? (
              <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in duration-700">
                <div className={`flex justify-between items-end border-b ${theme.border} pb-6`}>
                  <div><h2 className={`text-4xl font-black ${theme.textMain} tracking-tighter`}>{data.ticker}</h2><div className={`text-2xl font-bold ${theme.textMuted} mt-1`}>{data.current_price}</div></div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  <div className="space-y-6">
                    <div className={`${theme.card} rounded-3xl border ${theme.border} p-6 relative overflow-hidden shadow-sm`}>
                      <div className={`absolute top-0 left-0 w-full h-1 ${data.prediction === 'Bullish' ? 'bg-emerald-500' : 'bg-rose-500'}`}></div>
                      <p className={`${theme.textMuted} text-[10px] font-black uppercase mb-4 tracking-widest`}>AI Sentiment Verdict</p>
                      <div className="flex items-center justify-between mb-4"><span className={`text-4xl font-black ${data.prediction === 'Bullish' ? 'text-emerald-400' : 'text-rose-400'}`}>{data.prediction}</span>{data.prediction === 'Bullish' ? <TrendingUp className="text-emerald-500" size={32} /> : <TrendingDown className="text-rose-500" size={32} />}</div>
                      <div className={`${darkMode ? 'bg-slate-800/50' : 'bg-slate-100'} p-4 rounded-2xl`}><div className={`flex justify-between text-[10px] font-bold mb-2 ${theme.textMuted}`}><span>Confidence Score</span><span>{data.probability}%</span></div><div className={`w-full ${darkMode ? 'bg-slate-700' : 'bg-slate-200'} h-1.5 rounded-full overflow-hidden`}><div className="bg-emerald-500 h-full transition-all duration-1000" style={{width: `${data.probability}%`}}></div></div></div>
                    </div>
                    <div className={`${theme.card} rounded-3xl border ${theme.border} p-6 shadow-sm`}><p className={`${theme.textMuted} text-[10px] font-black uppercase mb-6 tracking-widest`}>Inference Catalysts</p><div className="space-y-5">{data.shap_values.map((v, i) => ( <div key={i}><div className="flex justify-between text-[11px] font-bold mb-2"><span className={theme.textMuted}>{v.feature}</span><span className={v.impact > 0 ? 'text-emerald-400' : 'text-rose-400 font-bold'}>{(v.impact * 100).toFixed(2)}%</span></div><div className={`w-full ${darkMode ? 'bg-slate-800' : 'bg-slate-100'} h-1 rounded-full overflow-hidden`}><div className={`h-full ${v.impact > 0 ? 'bg-emerald-500/40' : 'bg-rose-500/40'}`} style={{width: `${Math.min(Math.abs(v.impact * 1000), 100)}%`}}></div></div></div> ))}</div></div>
                  </div>

                  <div className={`${theme.card} lg:col-span-2 rounded-3xl border ${theme.border} p-8 flex flex-col min-h-[450px] shadow-sm`}>
                    <div className="flex justify-between items-center mb-10"><h3 className={`${theme.textMain} text-lg font-black tracking-tight`}>Price Action Projection</h3>
                      <div className="relative">
                        <select value={chartRange} onChange={(e) => setChartRange(e.target.value)} className={`appearance-none ${theme.input} ${theme.textMuted} text-[10px] font-black uppercase border ${theme.border} rounded-lg py-2 pl-4 pr-10 outline-none cursor-pointer transition-colors hover:border-emerald-500`}>
                          <option value="1Y">1 Year History</option><option value="1M">Last Month</option><option value="10D">10 Day Trend</option>
                        </select><ChevronDown className="absolute right-3 top-2.5 text-slate-500 pointer-events-none" size={14} />
                      </div>
                    </div>
                    <div className="flex-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={filteredTrendData}>
                          <defs><linearGradient id="colorP" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/><stop offset="95%" stopColor="#10b981" stopOpacity={0}/></linearGradient></defs>
                          <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? "#1e293b" : "#e2e8f0"} vertical={false} />
                          <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 10}} dy={10} interval={chartRange === '1Y' ? 40 : 2} />
                          <YAxis domain={['auto', 'auto']} axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 10}} />
                          <RechartsTooltip contentStyle={{backgroundColor: darkMode ? '#0f172a' : '#fff', border: `1px solid ${darkMode ? '#1e293b' : '#e2e8f0'}`, borderRadius: '12px'}} />
                          <Area type="monotone" dataKey="price" stroke="#10b981" strokeWidth={2} fill="url(#colorP)" animationDuration={1000} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-[60vh] flex flex-col items-center justify-center opacity-30 text-slate-500">
                <Activity size={64} strokeWidth={1} className="mb-4" />
                <p className="font-black text-xs uppercase tracking-[0.3em]">Command Center Offline</p>
                <p className="text-[10px] mt-2 font-bold uppercase tracking-widest">Initialize Terminal by Searching Ticker</p>
              </div>
            )
          ) : view === 'settings' ? (
            <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in duration-500">
               <h2 className={`text-3xl font-black ${theme.textMain} tracking-tighter`}>System Configuration</h2>
               <div className={`${theme.card} rounded-3xl border ${theme.border} p-8 shadow-lg space-y-10`}>
                  <div><h4 className={`text-xs font-black uppercase ${theme.textMuted} tracking-widest mb-6`}>Appearance</h4><div className={`flex items-center justify-between p-6 ${darkMode ? 'bg-emerald-500/5' : 'bg-slate-100'} border ${darkMode ? 'border-emerald-500/20' : 'border-slate-200'} rounded-2xl`}><div className="flex items-center space-x-4"><div className="p-3 bg-emerald-500/10 rounded-xl text-emerald-500">{darkMode ? <Moon size={24} /> : <Sun size={24} />}</div><div><p className={`text-sm font-bold ${theme.textMain}`}>Display Mode</p><p className={`text-[10px] ${theme.textMuted} font-medium`}>Switch between Quant Dark and Terminal Light themes.</p></div></div><button onClick={() => setDarkMode(!darkMode)} className={`relative w-14 h-7 rounded-full transition-colors duration-300 ${darkMode ? 'bg-emerald-500' : 'bg-slate-300'}`}><div className={`absolute top-1 w-5 h-5 bg-white rounded-full transition-all duration-300 ${darkMode ? 'right-1' : 'left-1'}`}></div></button></div></div>
                  <div><h4 className={`text-xs font-black uppercase ${theme.textMuted} tracking-widest mb-4`}>Memory Control</h4><button onClick={() => { if(window.confirm("Purge all data?")) { localStorage.clear(); window.location.reload(); }}} className="w-full flex items-center justify-between p-4 bg-rose-500/5 border border-rose-500/20 rounded-2xl text-rose-500 hover:bg-rose-500/10 transition-all font-bold"><div className="text-left"><p className="text-sm">Purge Local Memory</p><p className="text-[10px] opacity-60">Reset all theme and search preferences.</p></div><Trash2 size={18} /></button></div>
               </div>
            </div>
          ) : (
             <div className="max-w-6xl mx-auto animate-in fade-in duration-500">
                <h2 className={`text-3xl font-black ${theme.textMain} tracking-tighter mb-8`}>Market Leaderboard</h2>
                <div className={`${theme.card} rounded-3xl border ${theme.border} overflow-hidden shadow-sm`}>
                  <table className="w-full text-left"><thead className={`${darkMode ? 'bg-slate-800/40' : 'bg-slate-50'} ${theme.textMuted} text-[10px] font-black uppercase`}><tr><th className="px-8 py-6">Ticker</th><th className="px-8 py-6">Current Price</th><th className="px-8 py-6">Confidence</th><th className="px-8 py-6 text-right">Action</th></tr></thead>
                  <tbody className={`divide-y ${theme.border}`}>
                    {screenerData.map((s, i) => ( <tr key={i} className={`hover:${darkMode ? 'bg-slate-800/20' : 'bg-slate-50'} transition-all`}><td className={`px-8 py-6 font-black ${theme.textMain}`}>{s.ticker}</td><td className={`px-8 py-6 ${theme.textMuted}`}>{s.price}</td><td className="px-8 py-6 text-emerald-500 font-mono font-bold">{s.probability}%</td><td className="px-8 py-6 text-right"><button onClick={() => fetchAnalysis(s.ticker)} className={`text-[10px] ${darkMode ? 'bg-slate-800' : 'bg-slate-100'} px-4 py-2 rounded-xl font-black transition-all hover:bg-emerald-500 hover:text-white`}>TERMINAL</button></td></tr> ))}
                  </tbody></table>
                </div>
             </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;