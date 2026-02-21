import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { ExternalLink, Github, Menu, X, Check, Sparkles, ArrowRight, Zap, RefreshCw, PenTool, Mail, Calendar, FileText, HardDrive, MessageSquare, ChevronRight, Globe, Shield, Clock, Bot, Workflow, Star, Search, Code, Database, ShoppingBag, Image, Layers, LayoutTemplate, Briefcase, Volume2, ListTodo, ListChecks, Send } from 'lucide-react';
import { SignInButton, SignUpButton, UserButton, useUser, useAuth } from '@clerk/clerk-react';
import { useSetuAPI } from './services/api';
import SignInPrompt from './components/SignInPrompt';

const GMAIL_LOGO = 'https://www.gstatic.com/marketing-cms/assets/images/66/ac/14b165e647fd85c824bfbe5d6bc5/gmail.webp=s96-fcrop64=1,00000000ffffffff-rw';
const GDRIVE_LOGO = 'https://logos.composio.dev/api/googledrive';
const SLACK_LOGO = 'https://logos.composio.dev/api/slack';
const GSHEETS_LOGO = 'https://logos.composio.dev/api/googlesheets';
const NOTION_LOGO = 'https://logos.composio.dev/api/notion';
const GITHUB_LOGO = 'https://cdn.simpleicons.org/github/white';
const GMEET_LOGO = 'https://logos.composio.dev/api/googlemeet';
const GCALENDAR_LOGO = 'https://logos.composio.dev/api/googlecalendar';
const EXCEL_LOGO = 'https://cdn.jsdelivr.net/gh/ComposioHQ/open-logos@master/Excel.png';
const GDOCS_LOGO = 'https://logos.composio.dev/api/googledocs';
const DISCORD_LOGO = 'https://logos.composio.dev/api/discord';
const TELEGRAM_LOGO = 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Telegram_logo.svg/240px-Telegram_logo.svg.png';

const examples = [
    { trigger: "a new email arrives from a client", action: "create calendar reminder and add to Notion CRM" },
    { trigger: "a GitHub issue is created", action: "post it to #dev Slack with priority label" },
    { trigger: "someone mentions us on social media", action: "add it to our social tracker spreadsheet" },
    { trigger: "a support ticket arrives", action: "draft a reply using our FAQ doc" },
    { trigger: "a new lead is added to my sheet", action: "research the company and add notes" },
    { trigger: "a meeting is scheduled in Calendar", action: "create agenda doc and send to attendees" },
];

const tabs = ['Productivity', 'Communication', 'Development', 'Business', 'AI & More'];

const appsData = {
    'Popular Apps': [
        { name: 'Gmail', image: GMAIL_LOGO, color: '#EA4335', features: 13 },
        { name: 'Google Drive', image: GDRIVE_LOGO, color: '#0F9D58', features: 21 },
        { name: 'Slack', image: SLACK_LOGO, color: '#E01E5A', features: 20 },
        { name: 'Google Sheets', image: GSHEETS_LOGO, color: '#0F9D58', features: 23 },
        { name: 'Notion', image: NOTION_LOGO, color: '#FFFFFF', features: 28 },
        { name: 'GitHub', image: GITHUB_LOGO, color: '#FFFFFF', features: 17 },
        { name: 'Asana', icon: Check, color: '#F06A6A', features: 37 },
        { name: 'Salesforce', icon: Globe, color: '#00A1E0', features: 24 }
    ],
    'Productivity': [
        { name: 'Dropbox', icon: HardDrive, features: 11, beta: true },
        { name: 'Todoist', icon: Check, features: 14, beta: true },
        { name: 'Gmail', image: GMAIL_LOGO, features: 13, color: '#EA4335' },
        { name: 'Google Tasks', icon: Check, features: 14, beta: true, color: '#4285F4' },
        { name: 'OneDrive', icon: HardDrive, features: 13, beta: true, color: '#0078D4' },
        { name: 'Notion', image: NOTION_LOGO, features: 28 },
        { name: 'Outlook', icon: Mail, features: 20, beta: true, color: '#0078D4' },
        { name: 'Google Docs', image: GDOCS_LOGO, features: 26, beta: true, color: '#4285F4' },
        { name: 'api2pdf', icon: FileText, features: 5, beta: true },
        { name: 'Google Meet', image: GMEET_LOGO, features: 18, beta: true, color: '#00897B' },
        { name: 'Google Sheets', image: GSHEETS_LOGO, features: 23, beta: true, color: '#0F9D58' },
        { name: 'Google Calendar', image: GCALENDAR_LOGO, features: 31, beta: true, color: '#4285F4' },
    ],
    'Communication': [
        { name: 'Slack', image: SLACK_LOGO, features: 20, color: '#E01E5A' },
        { name: 'Discord', image: DISCORD_LOGO, features: 6, beta: true, color: '#5865F2' },
        { name: 'Twitter', icon: Globe, features: 28, beta: true, color: '#1DA1F2' },
        { name: 'Telegram', icon: Send, features: 18, beta: true, color: '#26A5E4' }
    ],
    'Development': [
        { name: 'AgentQL', icon: Code, features: 3, beta: true },
        { name: 'GitHub', image: GITHUB_LOGO, features: 17, beta: true },
        { name: 'Supabase', icon: Database, features: 31, beta: true, color: '#3ECF8E' },
        { name: 'GitLab', icon: Github, features: 28, beta: true, color: '#FC6D26' }
    ],
    'Business': [
        { name: 'Benchmark Email', icon: Mail, features: 13, beta: true },
        { name: 'Todoist', icon: Check, features: 14, beta: true },
        { name: 'Shopify', icon: ShoppingBag, features: 15, beta: true, color: '#96BF48' },
        { name: 'Google Tasks', icon: Check, features: 14, beta: true, color: '#4285F4' },
    ],
    'AI & More': [
        { name: 'Exa', icon: Zap, features: 12, beta: true },
        { name: 'Voo', icon: Zap, features: 5, beta: true },
        { name: 'Perplexity', icon: Search, features: 1, beta: true },
        { name: 'AtTest AI', icon: Check, features: 4, beta: true },
        { name: 'Canva', icon: Image, features: 28, beta: true, color: '#00C4CC' },
    ]
};


const SetuLogo = ({ size = 'lg' }) => (
    <div className="flex flex-col">
        <div className="flex items-center">
            <div className={`bg-dark-600 border border-white/10 rounded-lg flex items-center gap-1.5 ${size === 'lg' ? 'px-3 py-1.5' : 'px-2 py-1'}`}>
                <span className={`${size === 'lg' ? 'text-xl' : 'text-base'} font-black tracking-wider text-white`}>SETU</span>
                <span className={`${size === 'lg' ? 'text-lg' : 'text-sm'} font-bold text-accent`} style={{ fontFamily: 'serif' }}>सेतु</span>
            </div>
        </div>
        <span className="text-[10px] tracking-[0.15em] text-white/35 mt-1.5 italic">Bridge of your digital world</span>
    </div>
);

const AppCard = ({ app, onClick }) => {
    const Icon = app.icon;
    return (
        <div
            className="bg-[#1C1C1C] border border-white/5 hover:border-white/20 hover:bg-[#222222] rounded-xl p-4 flex items-center justify-between cursor-pointer transition-all group"
            onClick={onClick}
        >
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-white/5 border border-white/10 group-hover:bg-white/10 transition-colors">
                    {app.image ? (
                        <img src={app.image} alt={app.name} className="w-5 h-5 object-contain" />
                    ) : (
                        <Icon size={20} style={{ color: app.color || '#FFFFFF' }} />
                    )}
                </div>
                <div className="flex flex-col">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-white/90 group-hover:text-white transition-colors">{app.name}</span>
                        {app.beta && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-md bg-[#332211] text-[#FF8B5E] border border-[#FF6B35]/30">beta</span>
                        )}
                    </div>
                    <span className="text-xs text-white/40 group-hover:text-white/50 transition-colors">{app.features} features</span>
                </div>
            </div>
            <ChevronRight size={16} className="text-white/10 group-hover:text-white/30 transition-colors" />
        </div>
    )
};

const WorkflowNode = ({ label, icon: Icon, isImage, x, y, active, accent }) => (
    <div
        className="absolute flex flex-col items-center gap-2 transition-all duration-500 z-10"
        style={{ left: x, top: y, transform: 'translate(-50%, -50%)' }}
    >
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-300
            ${active
                ? 'bg-accent/20 border-accent/40 text-accent shadow-[0_0_15px_rgba(255,107,53,0.25)]'
                : 'bg-[#1C1C1C] border-white/10 text-white/70 shadow-lg'}
            ${accent ? 'ring-1 ring-accent/30' : ''}
        `}>
            {isImage ? (
                <img src={Icon} alt={label} className="w-6 h-6 object-contain" />
            ) : (
                <Icon size={22} />
            )}
        </div>
        <span className="text-[11px] font-medium text-white/70 whitespace-nowrap">{label}</span>
    </div>
);

const integrationsList = [
    { name: 'Gmail', image: GMAIL_LOGO, color: '#EA4335' },
    { name: 'Google Calendar', image: GCALENDAR_LOGO, color: '#4285F4' },
    { name: 'Notion', image: NOTION_LOGO, color: '#FFFFFF' },
    { name: 'Google Drive', image: GDRIVE_LOGO, color: '#0F9D58' },
    { name: 'GitHub', image: GITHUB_LOGO, color: '#FFFFFF' },
    { name: 'Slack', image: SLACK_LOGO, color: '#E01E5A' },
];

const SERVICE_CONFIG = {
    'Gmail': {
        color: '#EA4335',
        image: GMAIL_LOGO,
        description: 'Send emails, drafts, reminders',
        oauthType: 'google',
        examples: [
            'Send meeting reminder to john@test.com',
            'Draft a follow-up email to the client',
            'Send project update to team@company.com',
        ],
        placeholderText: 'e.g. Send reminder to john@test.com about the 3pm meeting'
    },
    'Google Calendar': {
        color: '#4285F4',
        image: GCALENDAR_LOGO,
        description: 'Create events, reminders, meetings',
        oauthType: 'google',
        examples: [
            'Set reminder Feb 20 at 2pm for team standup',
            'Schedule meeting tomorrow at 3pm with john@test.com',
            'Create weekly sync every Monday at 10am',
        ],
        placeholderText: 'e.g. Set reminder tomorrow at 2pm for team meeting'
    },
    'Google Meet': {
        color: '#00897B',
        image: GMEET_LOGO,
        description: 'Create meetings and send links',
        oauthType: 'google',
        examples: [
            'Schedule meeting tomorrow 3pm, send link to team@company.com',
            'Create standup at 9am every day and send link to 3 people',
        ],
        placeholderText: 'e.g. Create meeting tomorrow 3pm and send link to john@test.com'
    },
    'Google Drive': {
        color: '#0F9D58',
        image: GDRIVE_LOGO,
        description: 'Create, upload, share files',
        oauthType: 'google',
        examples: [
            'Create a folder called Q1 Planning',
            'Upload and share budget doc with team',
        ],
        placeholderText: 'e.g. Create a folder called Q1 Planning and share with team'
    },
    'Notion': {
        color: '#FFFFFF',
        image: NOTION_LOGO,
        description: 'Create pages, databases, notes',
        oauthType: 'notion',
        examples: [
            'Create project page for website redesign',
            'Add task to my todo list: Review PR #234',
            'Create meeting notes for today',
        ],
        placeholderText: 'e.g. Create a project page for Q1 planning'
    },
    'GitHub': {
        color: '#FFFFFF',
        image: GITHUB_LOGO,
        description: 'Create issues, PRs, manage repos',
        oauthType: 'github',
        examples: [
            'Create issue for login bug in myorg/myrepo',
            'Create PR description for feature/auth-fix',
        ],
        placeholderText: 'e.g. Create issue: Login bug in myorg/website repo'
    },
    'Slack': {
        color: '#E01E5A',
        image: SLACK_LOGO,
        description: 'Send messages, notifications',
        oauthType: 'slack',
        examples: [
            'Send message to #general: Deploy is done',
            'Notify #engineering about the new PR',
        ],
        placeholderText: 'e.g. Send message to #general: Meeting in 10 minutes'
    },
    'Google Sheets': {
        color: '#0F9D58',
        image: GSHEETS_LOGO,
        description: 'Create and update spreadsheets',
        oauthType: 'google',
        examples: [
            'Create budget spreadsheet for Q1',
            'Add row to expenses sheet: $500 for hosting',
        ],
        placeholderText: 'e.g. Create a budget spreadsheet for Q1 2026'
    },
    'Google Docs': {
        color: '#4285F4',
        image: GDOCS_LOGO,
        description: 'Create and edit documents',
        oauthType: 'google',
        examples: [
            'Create meeting agenda for tomorrow',
            'Draft project proposal for client',
        ],
        placeholderText: 'e.g. Create meeting agenda document for tomorrow'
    },
};

const toolsGrid = [
    { name: 'Gmail', image: GMAIL_LOGO, color: '#EA4335', bg: 'bg-red-500/10' },
    { name: 'Google Calendar', image: GCALENDAR_LOGO, color: '#4285F4', bg: 'bg-blue-500/10' },
    { name: 'Notion', image: NOTION_LOGO, color: '#FFFFFF', bg: 'bg-white/5' },
    { name: 'Google Drive', image: GDRIVE_LOGO, color: '#0F9D58', bg: 'bg-green-500/10' },
    { name: 'Slack', image: SLACK_LOGO, color: '#E01E5A', bg: 'bg-pink-500/10' },
    { name: 'GitHub', image: GITHUB_LOGO, color: '#FFFFFF', bg: 'bg-white/5' },
    { name: 'Google Sheets', image: GSHEETS_LOGO, color: '#0F9D58', bg: 'bg-green-500/10' },
    { name: 'Discord', image: DISCORD_LOGO, color: '#5865F2', bg: 'bg-indigo-500/10' },
    { name: 'Telegram', image: TELEGRAM_LOGO, color: '#0079BF', bg: 'bg-blue-500/10' },
    { name: 'Google Docs', image: GDOCS_LOGO, color: '#0052CC', bg: 'bg-blue-600/10' },
    { name: 'Excel', image: EXCEL_LOGO, color: '#5E6AD2', bg: 'bg-indigo-500/10' },
    { name: 'More', icon: Sparkles, color: '#FF6B35', bg: 'bg-accent/10' },
];

const Marquee = ({ items, reverse = false, className = '' }) => (
    <div className={`marquee-container ${className}`}>
        <div
            className="marquee-content"
            style={{
                animationDirection: reverse ? 'reverse' : 'normal',
                animationDuration: '40s'
            }}
        >
            {items.concat(items).concat(items).map((item, i) => (
                <div key={i} className="marquee-item">
                    <span className="text-xl mx-2">★</span>
                    {item}
                </div>
            ))}
        </div>
    </div>
);

const IntegrationPanel = ({
    serviceName,
    mode,
    taskInput,
    onTaskInputChange,
    onConnect,
    onRun,
    onClose,
    onRunAnother,
    result,
    isRunning,
    isConnected,
}) => {
    const config = SERVICE_CONFIG[serviceName];
    if (!config) return null;

    return (
        <>

            <div
                className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />


            <div className="fixed z-50 inset-x-0 bottom-0 sm:inset-auto sm:top-1/2 sm:left-1/2 sm:-translate-x-1/2 sm:-translate-y-1/2 w-full sm:w-[480px] bg-[#111] border border-white/10 rounded-t-2xl sm:rounded-2xl shadow-2xl overflow-hidden">


                <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
                    <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center">
                            {config.image ? (
                                <img src={config.image} alt={serviceName} className="w-5 h-5 object-contain" />
                            ) : null}
                        </div>
                        <div>
                            <p className="text-sm font-semibold text-white">{serviceName}</p>
                            <p className="text-[11px] text-white/40">{config.description}</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="w-7 h-7 rounded-lg bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors"
                    >
                        <X size={14} className="text-white/50" />
                    </button>
                </div>

                <div className="p-5">


                    {(mode === 'connect' || mode === 'connecting') && (
                        <div className="text-center py-4">
                            <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mx-auto mb-4">
                                {config.image && <img src={config.image} alt={serviceName} className="w-8 h-8 object-contain" />}
                            </div>
                            <p className="text-white font-semibold mb-1">Connect {serviceName}</p>
                            <p className="text-white/40 text-sm mb-6">
                                Allow SETU to access your {serviceName} account to automate tasks
                            </p>


                            <div className="text-left bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 mb-5 space-y-2">
                                <p className="text-[11px] text-white/40 uppercase tracking-wider mb-3">What SETU can do</p>
                                {config.examples.map((ex, i) => (
                                    <div key={i} className="flex items-center gap-2 text-sm text-white/60">
                                        <div className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
                                        {ex}
                                    </div>
                                ))}
                            </div>

                            <button
                                onClick={() => onConnect(serviceName)}
                                disabled={mode === 'connecting'}
                                className="w-full h-11 rounded-xl bg-accent hover:bg-[#dfff4d] text-black font-semibold text-sm transition-all flex items-center justify-center gap-2 disabled:opacity-50"
                            >
                                {mode === 'connecting' ? (
                                    <>
                                        <RefreshCw size={14} className="animate-spin" />
                                        Connecting...
                                    </>
                                ) : (
                                    <>Connect {serviceName} Account</>
                                )}
                            </button>
                        </div>
                    )}


                    {mode === 'task' && (
                        <div>

                            <div className="flex items-center gap-2 mb-4">
                                <div className="w-2 h-2 rounded-full bg-green-400" />
                                <span className="text-xs text-green-400 font-medium">{serviceName} connected</span>
                            </div>

                            <p className="text-white font-medium mb-3 text-sm">What do you want to do?</p>


                            <textarea
                                value={taskInput}
                                onChange={(e) => onTaskInputChange(e.target.value)}
                                placeholder={config.placeholderText}
                                rows={3}
                                className="w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-3 text-sm text-white placeholder:text-white/30 outline-none focus:border-accent/40 resize-none transition-colors"
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                                        onRun();
                                    }
                                }}
                            />
                            <p className="text-[11px] text-white/20 mt-1 mb-4">Press ⌘+Enter to run</p>


                            <div className="mb-5">
                                <p className="text-[11px] text-white/30 mb-2 uppercase tracking-wider">Quick examples</p>
                                <div className="flex flex-wrap gap-2">
                                    {config.examples.map((ex, i) => (
                                        <button
                                            key={i}
                                            onClick={() => onTaskInputChange(ex)}
                                            className="text-xs px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06] text-white/50 hover:text-white/80 hover:bg-white/[0.07] hover:border-white/10 transition-all text-left"
                                        >
                                            {ex}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <button
                                onClick={onRun}
                                disabled={!taskInput.trim()}
                                className="w-full h-11 rounded-xl bg-accent hover:bg-[#dfff4d] text-black font-semibold text-sm transition-all flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                <Sparkles size={14} className="fill-black" />
                                Run Agent
                            </button>
                        </div>
                    )}


                    {mode === 'running' && (
                        <div className="py-6 text-center">
                            <div className="w-14 h-14 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center mx-auto mb-4">
                                <RefreshCw size={22} className="text-accent animate-spin" />
                            </div>
                            <p className="text-white font-semibold mb-1">Agent is working...</p>
                            <p className="text-white/40 text-sm">{taskInput}</p>


                            <div className="mt-6 space-y-2 text-left">
                                {[
                                    'Analyzing your request with AI...',
                                    `Connecting to ${serviceName}...`,
                                    'Extracting actions from request...',
                                    'Executing workflow agents...'
                                ].map((step, i) => (
                                    <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02]">
                                        <RefreshCw size={12} className="text-accent animate-spin flex-shrink-0" style={{ animationDelay: `${i * 0.3}s` }} />
                                        <span className="text-xs text-white/50">{step}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}


                    {mode === 'result' && result && (
                        <div className="py-4">
                            <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4 ${result.success ? 'bg-green-500/10 border border-green-500/20' : 'bg-red-500/10 border border-red-500/20'}`}>
                                {result.success ? (
                                    <Check size={22} className="text-green-400" />
                                ) : (
                                    <X size={22} className="text-red-400" />
                                )}
                            </div>

                            <p className="text-center font-semibold text-white mb-1 text-sm">
                                {result.success ? 'Done!' : 'Something went wrong'}
                            </p>

                            {/* Show per-action results */}
                            {result.actionItems && result.actionItems.length > 0 ? (
                                <div className="mt-3 mb-5 space-y-2">
                                    {result.actionItems.map((item, idx) => (
                                        <div key={idx} className={`flex items-center gap-3 px-3 py-2.5 rounded-xl ${item.startsWith('✓') ? 'bg-green-500/5 border border-green-500/10' : 'bg-red-500/5 border border-red-500/10'}`}>
                                            {item.startsWith('✓') ? (
                                                <Check size={14} className="text-green-400 flex-shrink-0" />
                                            ) : (
                                                <X size={14} className="text-red-400 flex-shrink-0" />
                                            )}
                                            <span className="text-xs text-white/70">{item.replace(/^[✓✗]\s*/, '')}</span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-center text-white/50 text-sm mb-5">{result.message}</p>
                            )}


                            <div className="flex gap-3">
                                <button
                                    onClick={onRunAnother}
                                    className="flex-1 h-10 rounded-xl bg-white/[0.05] hover:bg-white/[0.08] text-white/70 text-sm font-medium transition-all"
                                >
                                    Run another
                                </button>
                                <button
                                    onClick={onClose}
                                    className="flex-1 h-10 rounded-xl bg-accent hover:bg-[#dfff4d] text-black text-sm font-semibold transition-all"
                                >
                                    Done
                                </button>
                            </div>
                        </div>
                    )}

                </div>
            </div>
        </>
    );
};

export default function SetuLandingPage() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);
    const [isAppsModalOpen, setIsAppsModalOpen] = useState(false);
    const [activeTab, setActiveTab] = useState('Productivity');
    const [currentExampleIndex, setCurrentExampleIndex] = useState(0);
    const [isTransitioning, setIsTransitioning] = useState(false);

    const { isSignedIn, isLoaded } = useAuth();
    const { user } = useUser();
    const setuAPI = useSetuAPI();
    const [taskInput, setTaskInput] = useState('');
    const [isCreatingTask, setIsCreatingTask] = useState(false);
    const [taskStatus, setTaskStatus] = useState(null);
    const [showSignInPrompt, setShowSignInPrompt] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [integrations, setIntegrations] = useState([]);

    const [activeIntegration, setActiveIntegration] = useState(null);
    const [panelMode, setPanelMode] = useState('connect');
    const [connectedServices, setConnectedServices] = useState([]);
    const [agentTaskInput, setAgentTaskInput] = useState('');
    const [agentResult, setAgentResult] = useState(null);
    const [agentRunning, setAgentRunning] = useState(false);

    useEffect(() => {
        setTimeout(() => setIsLoading(false), 1000);
    }, []);

    const [searchParams, setSearchParams] = useSearchParams();
    useEffect(() => {
        const integration = searchParams.get('integration');
        const status = searchParams.get('status');
        if (status === 'success' && integration) {
            setTaskStatus(`✓ Successfully connected ${integration.charAt(0).toUpperCase() + integration.slice(1)}!`);
            // Optional: Clear params after reading
            // setSearchParams({}); 
        }
    }, [searchParams]);


    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });


    useEffect(() => {
        const cursor = document.createElement('div');
        cursor.className = 'custom-cursor';
        document.body.appendChild(cursor);


        const handleMouseMove = (e) => {
            cursor.style.left = e.clientX + 'px';
            cursor.style.top = e.clientY + 'px';
        };


        const handleMouseEnter = () => cursor.classList.add('hover');
        const handleMouseLeave = () => cursor.classList.remove('hover');


        const handleMouseDown = () => cursor.classList.add('click');
        const handleMouseUp = () => cursor.classList.remove('click');

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mouseup', handleMouseUp);


        const interactiveElements = document.querySelectorAll('a, button, [role="button"], .card, .glass-card, .marquee-item');
        interactiveElements.forEach(el => {
            el.addEventListener('mouseenter', handleMouseEnter);
            el.addEventListener('mouseleave', handleMouseLeave);
        });

        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mouseup', handleMouseUp);
            interactiveElements.forEach(el => {
                el.removeEventListener('mouseenter', handleMouseEnter);
                el.removeEventListener('mouseleave', handleMouseLeave);
            });
            cursor.remove();
        };
    }, []);


    useEffect(() => {
        const handleMouseMove = (e) => {
            setMousePosition({
                x: (e.clientX / window.innerWidth - 0.5) * 20,
                y: (e.clientY / window.innerHeight - 0.5) * 20
            });
        };

        window.addEventListener('mousemove', handleMouseMove);
        return () => window.removeEventListener('mousemove', handleMouseMove);
    }, []);


    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 0);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);


    useEffect(() => {
        const interval = setInterval(() => {
            setIsTransitioning(true);
            setTimeout(() => {
                setCurrentExampleIndex((prev) => (prev + 1) % examples.length);
                setIsTransitioning(false);
            }, 400);
        }, 4000);
        return () => clearInterval(interval);
    }, []);


    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const success = urlParams.get('success');

        if (success) {

            const serviceMap = {
                'google_connected': ['Gmail', 'Google Calendar', 'Google Meet', 'Google Drive', 'Google Sheets', 'Google Docs'],
                'notion_connected': ['Notion'],
                'github_connected': ['GitHub'],
                'slack_connected': ['Slack'],
            };

            const connectedList = serviceMap[success] || [];
            setConnectedServices(prev => [...new Set([...prev, ...connectedList])]);

            // Read the saved service from localStorage (set before OAuth redirect)
            const savedService = localStorage.getItem('setu_pending_service');
            console.log('[SETU OAuth] success:', success, 'savedService:', savedService);

            if (savedService && SERVICE_CONFIG[savedService]) {
                setActiveIntegration(savedService);
                setPanelMode('task');
                localStorage.removeItem('setu_pending_service');
            }

            loadIntegrations();

            setTimeout(() => window.history.replaceState({}, '', '/'), 100);
        }
    }, []);


    const loadIntegrations = useCallback(async () => {
        if (isSignedIn && user?.id) {
            setuAPI.getIntegrations().then(data => {
                setIntegrations(data);


                const connected = [];
                if (data.google_connected) {
                    connected.push('Gmail', 'Google Calendar', 'Google Meet', 'Google Drive', 'Google Sheets', 'Google Docs');
                }
                if (data.notion_connected) connected.push('Notion');
                if (data.github_connected) connected.push('GitHub');
                if (data.slack_connected) connected.push('Slack');

                setConnectedServices(connected);
            }).catch(() => { });
        }
    }, [user?.id, isSignedIn]);


    useEffect(() => {
        loadIntegrations();
    }, [loadIntegrations]);

    const handleCreateTask = async () => {
        if (!isSignedIn) {
            setShowSignInPrompt(true);
            return;
        }

        setIsCreatingTask(true);
        setTaskStatus(null);

        try {
            const taskDescription = taskInput || examples[currentExampleIndex].action;
            const response = await setuAPI.createTask(taskDescription);
            console.log('Task created:', response);
            setTaskStatus('✓ Task created! Agents processing...');
            setTaskInput(''); // Clear input after successful creation
        } catch (error) {
            console.error('Failed to create task:', error);
            setTaskStatus('Error: Failed to create task. Please try again.');
        } finally {
            setIsCreatingTask(false);
        }
    };

    const handleIntegrationClick = (serviceName) => {
        if (!isSignedIn) {
            setShowSignInPrompt(true);
            return;
        }

        setActiveIntegration(serviceName);
        setAgentTaskInput('');
        setAgentResult(null);

        const isConnected = connectedServices.includes(serviceName);
        setPanelMode(isConnected ? 'task' : 'connect');
    };

    const handleClosePanel = () => {
        setActiveIntegration(null);
        setPanelMode('connect');
        setAgentTaskInput('');
        setAgentResult(null);
        setAgentRunning(false);
    };

    const handleOAuthConnect = async (serviceName) => {
        const config = SERVICE_CONFIG[serviceName];
        if (!config) return;

        setPanelMode('connecting');

        try {
            const oauthType = config.oauthType;

            localStorage.setItem('setu_pending_service', serviceName);

            if (oauthType === 'google') {
                const data = await setuAPI.initiateGoogleAuth();
            } else if (oauthType === 'notion') {
                const data = await setuAPI.initiateNotionAuth();
            } else if (oauthType === 'github') {
                const data = await setuAPI.initiateGitHubAuth();
            } else if (oauthType === 'slack') {
                const data = await setuAPI.initiateSlackAuth();
            }
        } catch (err) {
            console.error('OAuth failed:', err);
            setPanelMode('connect');
        }
    };

    const handleRunAgentTask = async () => {
        if (!agentTaskInput.trim()) return;

        setAgentRunning(true);
        setPanelMode('running');
        setAgentResult(null);

        try {
            const response = await setuAPI.createTask(agentTaskInput, 'high', activeIntegration);

            if (response.status === 'completed') {
                const actionItems = response.action_items || [];
                setAgentResult({
                    success: true,
                    message: formatResult(response, activeIntegration),
                    actionItems: actionItems.length > 1 ? actionItems : null,
                    data: response
                });
            } else if (response.status === 'failed') {
                const actionItems = response.action_items || [];
                const errorMsg = actionItems[0] || 'Task failed. Check your API keys and connected accounts.';
                setAgentResult({
                    success: false,
                    message: errorMsg,
                    actionItems: actionItems.length > 0 ? actionItems : null,
                });
            } else {
                setAgentResult({
                    success: true,
                    message: 'Task submitted successfully!'
                });
            }

            setPanelMode('result');
            setAgentRunning(false);

        } catch (err) {
            console.error('Task failed:', err);

            let message = err.message || 'Failed to run task';
            if (message.includes('not connected')) {
                message = `❌ ${activeIntegration} is not connected. Click "Connect" to link your account first.`;
            } else if (message.includes('credentials')) {
                message = `❌ Authentication failed. Try reconnecting ${activeIntegration}.`;
            } else if (message.includes('must be signed in')) {
                message = '❌ Please sign in first.';
            }

            setAgentResult({
                success: false,
                message
            });
            setPanelMode('result');
            setAgentRunning(false);
        }
    };


    const formatResult = (taskStatus, service) => {
        if (!taskStatus) return '✓ Task completed!';

        if (taskStatus.status === 'failed') {
            const err = taskStatus.action_items?.[0] || 'Task failed';
            return err;
        }

        const actionItems = taskStatus.action_items || [];

        // Multi-action summary
        if (actionItems.length > 1) {
            const successCount = actionItems.filter(item => item.startsWith('✓')).length;
            return `${successCount} of ${actionItems.length} actions completed`;
        }

        const successItem = actionItems.find(item => item.startsWith('✓'));
        if (successItem) return successItem;

        const serviceMessages = {
            'Gmail': '✓ Email sent successfully!',
            'Google Calendar': '✓ Calendar event created!',
            'Google Meet': '✓ Meeting created with Meet link!',
            'Google Drive': '✓ File created in Drive!',
            'Google Docs': '✓ Document created!',
            'Google Sheets': '✓ Spreadsheet created!',
            'Notion': '✓ Notion page created!',
            'GitHub': '✓ GitHub issue created!',
            'Slack': '✓ Slack message sent!',
        };

        return serviceMessages[service] || '✓ Task completed successfully!';
    };

    return (
        <div className="min-h-screen bg-dark-900 text-white selection:bg-accent selection:text-black overflow-x-hidden font-sans">
            {activeIntegration && (
                <IntegrationPanel
                    serviceName={activeIntegration}
                    mode={panelMode}
                    taskInput={agentTaskInput}
                    onTaskInputChange={setAgentTaskInput}
                    onConnect={handleOAuthConnect}
                    onRun={handleRunAgentTask}
                    onClose={handleClosePanel}
                    onRunAnother={() => {
                        setPanelMode('task');
                        setAgentResult(null);
                        setAgentTaskInput('');
                        setAgentRunning(false);
                    }}
                    result={agentResult}
                    isRunning={agentRunning}
                    isConnected={connectedServices.includes(activeIntegration)}
                />
            )}

            {isLoading && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-dark-900">
                    <div className="text-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent mx-auto mb-4"></div>
                        <p className="text-white/60">Loading SETU...</p>
                    </div>
                </div>
            )}

            <SignInPrompt show={showSignInPrompt} onClose={() => setShowSignInPrompt(false)} />

            <div className="grain-overlay"></div>

            <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <div className="animated-line" style={{ left: '15%', animationDelay: '0s' }} />
                <div className="animated-line" style={{ left: '45%', animationDelay: '4s' }} />
                <div className="animated-line" style={{ left: '75%', animationDelay: '8s' }} />
            </div>

            <div className="hero-glow fixed inset-0 pointer-events-none z-0 opacity-40" />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[600px] rounded-full opacity-10 blur-[120px] bg-accent/20 pointer-events-none" />

            <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-12 lg:px-20 py-2 bg-dark-950/80 backdrop-blur-md border-b border-white/[0.05]">
                <div className="flex items-center gap-3">
                    <SetuLogo size="sm" />
                    <span className="hidden sm:inline-block text-[10px] tracking-widest font-semibold text-white/30 uppercase border border-white/10 px-2 py-0.5 rounded-full">Early Access</span>
                </div>


                <div className="hidden md:flex items-center bg-white/[0.03] border border-white/[0.06] rounded-full px-1 py-1 backdrop-blur-md">
                    {['About', 'Product', 'Prices'].map((item) => (
                        <a
                            key={item}
                            href="#"
                            className="px-3 py-1.5 text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.05] rounded-full transition-all duration-200"
                        >
                            {item}
                        </a>
                    ))}
                    <div className="w-px h-3 bg-white/10 mx-1"></div>
                    <a href="#" className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-white/60 hover:text-white hover:bg-white/[0.05] rounded-full transition-all duration-200">
                        <Github size={14} />
                        Community
                    </a>
                </div>


                <div className="hidden md:flex items-center gap-6">
                    <a href="#how-it-works" className="text-xs font-medium text-white/50 hover:text-white transition-colors flex items-center gap-1 group"
                        onClick={(e) => { e.preventDefault(); document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' }); }}
                    >
                        How it works
                        <ExternalLink size={10} className="group-hover:-translate-y-0.5 group-hover:translate-x-0.5 transition-transform" />
                    </a>

                    {!isLoaded ? (
                        <div className="w-8 h-8 rounded-full bg-white/10 animate-pulse"></div>
                    ) : isSignedIn ? (
                        <div className="flex items-center gap-3">
                            <span className="text-xs text-white/60">
                                Hey, {user?.firstName || 'there'}!
                            </span>
                            <UserButton
                                afterSignOutUrl="/"
                                appearance={{
                                    elements: {
                                        avatarBox: "w-8 h-8",
                                        userButtonPopoverCard: "bg-dark-900 border border-white/10",
                                        userButtonPopoverActionButton: "hover:bg-white/5 text-white/80",
                                        userButtonPopoverActionButtonText: "text-white/70",
                                        userButtonPopoverFooter: "hidden"
                                    }
                                }}
                            />
                        </div>
                    ) : (
                        <div className="flex items-center gap-3">
                            <SignInButton mode="modal">
                                <button className="text-xs font-medium text-white/70 hover:text-white transition-colors">
                                    Sign In
                                </button>
                            </SignInButton>
                            <SignUpButton mode="modal">
                                <button className="bg-accent hover:bg-accent/90 text-black text-xs font-semibold px-4 py-1.5 rounded-lg transition-all shadow-[0_0_15px_rgba(184,255,0,0.3)] hover:shadow-[0_0_20px_rgba(184,255,0,0.5)]">
                                    Get Started
                                </button>
                            </SignUpButton>
                        </div>
                    )}
                </div>


                <button
                    className="md:hidden text-white/70"
                    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                >
                    {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </nav>


            {mobileMenuOpen && (
                <div className="md:hidden fixed inset-0 z-40 bg-dark-900/95 backdrop-blur-lg pt-20 px-6">
                    <div className="flex flex-col gap-6">
                        {['About', 'Product', 'Prices', 'How it works'].map((item) => (
                            <a key={item} href="#" className="text-lg text-white/70 hover:text-white transition-colors">
                                {item}
                            </a>
                        ))}

                        {!isLoaded ? (
                            <div className="w-full h-12 bg-white/10 rounded-lg animate-pulse mt-4"></div>
                        ) : isSignedIn ? (
                            <div className="flex items-center gap-4 mt-4 p-4 bg-white/5 rounded-lg">
                                <UserButton afterSignOutUrl="/" />
                                <span className="text-white/80">{user?.firstName || user?.emailAddresses[0].emailAddress}</span>
                            </div>
                        ) : (
                            <>
                                <SignInButton mode="modal">
                                    <button className="text-lg text-white/70 hover:text-white transition-colors text-left">
                                        Sign In
                                    </button>
                                </SignInButton>
                                <SignUpButton mode="modal">
                                    <button className="bg-accent text-white font-semibold px-6 py-3 rounded-lg w-full mt-4">
                                        Get Started
                                    </button>
                                </SignUpButton>
                            </>
                        )}
                    </div>
                </div>
            )}


            <section className="relative z-10 px-6 md:px-12 lg:px-20 pt-24 md:pt-28 pb-10 md:pb-16">
                <div
                    className="max-w-4xl mx-auto text-center"
                    style={{
                        transform: `translate(${mousePosition.x}px, ${mousePosition.y}px)`,
                        transition: 'transform 0.3s ease-out'
                    }}
                >

                    <h1 className="font-serif text-3xl sm:text-4xl md:text-[2.75rem] font-medium text-white tracking-[-0.02em] font-playfair flex flex-col items-center gap-2 md:gap-3">
                        <span>Describe any task in</span>

                        <span className="relative inline-block whitespace-nowrap">
                            <span className="italic text-accent">plain English</span>.

                            <svg
                                className="absolute left-0 w-full h-[6px] md:h-[8px] -bottom-1 md:-bottom-2 text-accent opacity-90"
                                viewBox="0 0 200 12"
                                fill="none"
                                preserveAspectRatio="none"
                            >
                                <path
                                    d="M3 9C50 3 150 3 197 9"
                                    stroke="currentColor"
                                    strokeWidth="3.5"
                                    strokeLinecap="round"
                                    vectorEffect="non-scaling-stroke"
                                />
                            </svg>
                        </span>

                        <span>Our AI agents bridge Gmail,Calendar,</span>
                        <span>and Notion, etc — automatically.</span>
                    </h1>


                    <p className="mt-4 text-white/45 text-sm md:text-base max-w-lg mx-auto leading-relaxed">
                        Setu is a no-code AI agent platform that helps you automate your daily tasks.
                    </p>


                    <div className="mt-8 max-w-2xl mx-auto">

                        <div className="relative rounded-2xl p-px bg-gradient-to-b from-white/[0.12] to-white/[0.04]">
                            <div className="rounded-2xl overflow-hidden bg-[#1d1d1d]">
                                <div className="px-3 py-3 pb-3 space-y-[4px]">


                                    <div className="rounded-xl bg-[#262626] px-4 py-3 border border-white/[0.06] transition-all duration-200 hover:border-white/[0.12] focus-within:border-white/[0.18] focus-within:bg-[#2a2a2a] cursor-text">
                                        <div className="flex items-start gap-3">
                                            <span className="text-[#6b7280] text-[14px] font-medium shrink-0 pt-0.5">When</span>
                                            <div className="flex-1 relative">
                                                <div className={`w-full bg-transparent outline-none text-[14px] z-10 relative min-h-[24px] text-left mt-[1px] text-[#9ca3af] ${isTransitioning ? 'animate-fade-out-up' : 'animate-fade-in-up'}`}>
                                                    {examples[currentExampleIndex].trigger}
                                                </div>
                                            </div>
                                            <PenTool size={14} className="text-white/20 shrink-0 mt-1" />
                                        </div>
                                    </div>


                                    <div className="rounded-xl bg-[#262626] px-4 py-3 border border-white/[0.06] transition-all duration-200 hover:border-white/[0.12] focus-within:border-white/[0.18] focus-within:bg-[#2a2a2a] cursor-text">
                                        <div className="flex items-start gap-3">
                                            <span className="text-accent text-[14px] font-medium shrink-0 pt-0.5 flex items-center gap-1">
                                                Automatically <span className="text-accent/60">→</span>
                                            </span>
                                            <div className="flex-1 relative">
                                                <div
                                                    className="w-full bg-transparent outline-none text-[14px] z-10 relative min-h-[24px] text-left mt-[1px] text-white/90"
                                                    contentEditable={true}
                                                    onInput={(e) => setTaskInput(e.currentTarget.textContent)}
                                                    suppressContentEditableWarning
                                                >
                                                    {taskInput || (isTransitioning
                                                        ? ''
                                                        : examples[currentExampleIndex].action
                                                    )}
                                                </div>
                                            </div>
                                            <PenTool size={14} className="text-white/20 shrink-0 mt-1" />
                                        </div>
                                    </div>


                                    <div className="flex items-center justify-between pt-1.5 px-1">
                                        <div className="flex items-center gap-1.5">

                                            <button
                                                onClick={() => setIsAppsModalOpen(true)}
                                                className="bg-white/[0.04] hover:bg-white/[0.07] text-[#a1a1aa] hover:text-white px-3 h-[38px] rounded-xl transition-all duration-150 flex items-center gap-2 text-[13px]"
                                            >
                                                <div className="flex items-center">
                                                    <div className="w-[18px] h-[18px] bg-white rounded-full flex items-center justify-center shadow-sm z-30">
                                                        <img src={SLACK_LOGO} alt="Slack" className="w-[10px] h-[10px]" />
                                                    </div>
                                                    <div className="w-[18px] h-[18px] bg-white rounded-full flex items-center justify-center shadow-sm -ml-1.5 z-20">
                                                        <img src={GSHEETS_LOGO} alt="Sheets" className="w-[10px] h-[10px]" />
                                                    </div>
                                                    <div className="w-[18px] h-[18px] bg-white rounded-full flex items-center justify-center shadow-sm -ml-1.5 z-10">
                                                        <img src={NOTION_LOGO} alt="Notion" className="w-[10px] h-[10px]" />
                                                    </div>
                                                </div>
                                                <span className="hidden sm:inline font-medium">Integrations</span>
                                            </button>


                                            <button className="bg-white/[0.04] hover:bg-white/[0.07] text-[#a1a1aa] hover:text-white px-3 h-[38px] rounded-xl transition-all duration-150 flex items-center gap-1.5 text-[13px] font-medium">
                                                <Zap size={14} className="fill-current" />
                                                <span className="hidden sm:inline">When something happens</span>
                                            </button>
                                        </div>


                                        <button
                                            onClick={handleCreateTask}
                                            disabled={isCreatingTask}
                                            className={`relative overflow-hidden group/btn text-black h-[38px] px-4 rounded-xl text-[13px] font-semibold flex items-center gap-1.5 transition-all duration-200 active:scale-[0.98] ${isCreatingTask
                                                ? 'bg-white/10 text-white/40 cursor-not-allowed'
                                                : 'bg-accent hover:bg-[#dfff4d] hover:shadow-lg hover:shadow-accent/20'
                                                }`}
                                        >
                                            {isCreatingTask ? (
                                                <RefreshCw size={14} className="animate-spin" />
                                            ) : (
                                                <Sparkles size={14} className="fill-black" />
                                            )}
                                            <span>{isCreatingTask ? 'Creating...' : 'Create'}</span>
                                        </button>
                                    </div>

                                    {taskStatus && (
                                        <div className={`mt-2 mx-3 mb-3 text-xs font-medium px-3 py-2 rounded-lg ${taskStatus.includes('✓')
                                            ? 'bg-accent/10 text-accent border border-accent/20'
                                            : 'bg-red-500/10 text-red-400 border border-red-500/20'
                                            }`}>
                                            {taskStatus}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>


                <div className="mt-12 flex flex-col sm:flex-row items-center justify-center gap-4">
                    <button className="neon-button text-base px-8 py-4 rounded-lg w-full sm:w-auto min-w-[200px] group flex items-center justify-center gap-2">
                        Start Automating Free
                        <span className="group-hover:translate-x-1 transition-transform">→</span>
                    </button>
                    <a
                        href="https://phase-feverfew-5ae.notion.site/SETU-AI-AGENT-WORKFLOW-AUTOMATION-PLATFORM-30b2ce28a6248017892ee3398517c8e8"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.08] text-white/90 text-sm font-medium transition-all border border-white/10"
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-white/80">
                            <path d="M8 5V19L19 12L8 5Z" fill="currentColor" />
                        </svg>
                        Documentation
                    </a>
                </div>


                <div className="mt-24 -mx-6 md:-mx-12 lg:-mx-20 transform -rotate-1 origin-left border-y-2 border-black">
                    <Marquee
                        items={['✓ AI AGENTS', '★ AUTOMATION', '✓ NO-CODE', '★ INTEGRATIONS', '✓ WORKFLOWS', '★ PRODUCTIVITY']}
                        className="bg-[#CBFF00] text-black py-4"
                    />
                </div>
            </section>


            <div className="relative z-20 py-0 bg-[#0a0a0a] border-y border-white/10">
                <Marquee
                    items={['★ GMAIL', '★ NOTION', '★ SLACK', '★ LINEAR', '★ DISCORD', '★ GITHUB', '★ HUBSPOT']}
                    reverse
                    className="bg-[#1a1a1a] text-white py-4"
                />
            </div>

            {/* ══════════════════ NOT SURE SECTION ══════════════════ */}
            <div className="max-w-6xl mx-auto px-6 pt-16 md:pt-28">
                <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            </div>
            <section className="relative z-10 text-center px-6 py-16 md:py-24">
            </section>


            <section id="how-it-works" className="relative z-10 max-w-6xl mx-auto px-6 pb-16 scroll-mt-20">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-5">


                    <div className="glass-card p-5">
                        <h3 className="text-[11px] tracking-[0.15em] uppercase text-white/40 font-semibold mb-4">Integrations</h3>
                        <div className="space-y-2.5">
                            {integrationsList.map((item) => {
                                const isConnected = connectedServices.includes(item.name);
                                return (
                                    <div
                                        key={item.name}
                                        className="flex items-center justify-between px-3 py-2.5 rounded-xl bg-white/[0.03] hover:bg-white/[0.06] border border-white/[0.04] hover:border-white/10 transition-all duration-200 group cursor-pointer"
                                        onClick={() => handleIntegrationClick(item.name)}
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-white/[0.06]">
                                                {item.image ? (
                                                    <img src={item.image} alt={item.name} className="w-4 h-4 object-contain" />
                                                ) : (
                                                    <item.icon size={16} style={{ color: item.color }} />
                                                )}
                                            </div>
                                            <span className="text-sm text-white/80 font-medium">{item.name}</span>
                                        </div>
                                        {isConnected ? (
                                            <div className="flex items-center gap-1.5">
                                                <div className="w-2 h-2 rounded-full bg-green-400" />
                                                <span className="text-[10px] text-green-400 font-medium">Connected</span>
                                            </div>
                                        ) : (
                                            <ChevronRight size={14} className="text-white/20 group-hover:text-white/50 transition-colors" />
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>


                    <div className="glass-card p-5 relative overflow-hidden min-h-[340px]">
                        <h3 className="text-[11px] tracking-[0.15em] uppercase text-white/40 font-semibold mb-4">Live Workflow</h3>


                        <div className="relative h-[280px]">

                            <svg className="absolute inset-0 w-full h-full z-0" xmlns="http://www.w3.org/2000/svg">
                                <defs>
                                    <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                        <stop offset="0%" stopColor="#B8FF00" stopOpacity="0.8" />
                                        <stop offset="100%" stopColor="#B8FF00" stopOpacity="0.2" />
                                    </linearGradient>
                                </defs>



                                <path d="M 78 134 L 125 134" stroke="#B8FF00" strokeOpacity="0.5" strokeWidth="2" strokeDasharray="5 4" fill="none" />



                                <path d="M 173 134 C 200 134, 200 64, 227 64" stroke="#B8FF00" strokeOpacity="0.5" strokeWidth="2" strokeDasharray="5 4" fill="none" />



                                <path d="M 173 134 C 200 134, 200 204, 227 204" stroke="#B8FF00" strokeOpacity="0.5" strokeWidth="2" strokeDasharray="5 4" fill="none" />
                            </svg>



                            <div className="absolute flex flex-col items-center gap-2 z-10" style={{ left: '54px', top: '110px', transform: 'translateX(-50%)' }}>
                                <div className="w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-500 bg-white/[0.04] border-white/15 text-white/70">
                                    <Volume2 size={20} />
                                </div>
                                <span className="text-[10px] font-medium text-white/70 whitespace-nowrap">One Prompt</span>
                            </div>


                            <div className="absolute flex flex-col items-center gap-2 z-10" style={{ left: '149px', top: '110px', transform: 'translateX(-50%)' }}>
                                <div className="w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-500 bg-white/[0.03] border-white/10 text-white/70">
                                    <ListChecks size={20} />
                                </div>
                                <span className="text-[10px] font-medium text-white/70 text-center leading-tight">Multi-Agent<br />Execution</span>
                            </div>


                            <div className="absolute flex flex-col items-center gap-2 z-10" style={{ left: '251px', top: '40px', transform: 'translateX(-50%)' }}>
                                <div className="w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-500 bg-[#1A1A1A] border-white/10 text-white/70 overflow-hidden">
                                    <img src={NOTION_LOGO} alt="Notion" className="w-6 h-6 object-contain opacity-90" />
                                </div>
                                <span className="text-[10px] font-medium text-white/70 whitespace-nowrap">Notion Updater</span>
                            </div>


                            <div className="absolute flex flex-col items-center gap-2 z-10" style={{ left: '251px', top: '180px', transform: 'translateX(-50%)' }}>
                                <div className="w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-500 bg-[#1A1A1A] border-white/10 text-white/70 overflow-hidden">
                                    <img src={GMAIL_LOGO} alt="Gmail" className="w-6 h-6 object-contain" />
                                </div>
                                <span className="text-[10px] font-medium text-white/70 whitespace-nowrap">Email Drafter</span>
                            </div>
                        </div>
                    </div>


                    <div className="glass-card p-5">
                        <h3 className="text-[11px] tracking-[0.15em] uppercase text-white/40 font-semibold mb-4">Agent Output Preview</h3>

                        <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                            <p className="text-xs text-white/40 mb-2 font-medium">Drafted email</p>
                            <div className="space-y-3 text-[13px] text-white/70 leading-relaxed">
                                <p>I want you to received your items in good condition after dispatch from our warehouse.</p>
                                <p>Please contact us immediately if you encounter any issues with packaging and delivery.</p>
                                <p>Thank you for your hectic active business.</p>
                                <p className="text-white/40 mt-1">Sincerely,</p>
                                <p className="text-white/40">Rushi.</p>
                            </div>
                            <div className="flex items-center gap-2 mt-5">
                                <button className="bg-accent/90 hover:bg-accent text-white text-xs font-semibold px-4 py-2 rounded-lg transition-all flex items-center gap-1.5">
                                    <Check size={12} />
                                    Approve & Send
                                </button>
                                <button className="bg-white/[0.06] hover:bg-white/10 text-white/60 hover:text-white/80 text-xs font-medium px-4 py-2 rounded-lg transition-all border border-white/[0.06]">
                                    Edit
                                </button>
                            </div>
                        </div>
                    </div>
                </div>


                <div className="mt-6 glass-card p-5">
                    <div className="flex items-center gap-2 mb-3">
                        <div className="w-3 h-3 rounded-full bg-red-500/60" />
                        <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
                        <div className="w-3 h-3 rounded-full bg-green-500/60" />
                        <span className="ml-2 text-[11px] text-white/30 font-mono">setu-agent</span>
                    </div>
                    <div className="font-mono text-xs md:text-sm space-y-1.5 text-white/60">
                        <p><span className="text-green-400">›</span> <span className="text-accent/80">Result:</span> Checking calendar for conflicts...</p>
                        <p><span className="text-green-400">›</span> <span className="text-accent/80">Result:</span> <span className="text-white/40">google_calendar.get_events("next Tuesday")...</span></p>
                        <p><span className="text-green-400">›</span> <span className="text-green-400/80">✓</span> No conflicts found. Meeting scheduled successfully.</p>
                        <p className="flex items-center"><span className="text-green-400">›</span>&nbsp;<span className="cursor-blink text-accent">▊</span></p>
                    </div>
                </div>
            </section>


            <section className="relative z-10 py-16 md:py-24">

                <div className="max-w-6xl mx-auto px-6 mb-16">
                    <div className="h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                </div>

                <div className="text-center px-6 mb-12">
                    <p className="text-xs tracking-[0.2em] uppercase text-white/30 mb-3">Connect your favorite tools</p>
                    <h2 className="text-2xl md:text-4xl font-bold">Connect your favorite tools</h2>
                    <p className="text-white/50 mt-3 text-sm md:text-base max-w-lg mx-auto">
                        Connect Setu agents to the workflow apps you already use.
                    </p>
                </div>

                <div className="max-w-5xl mx-auto px-6">
                    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-4 md:gap-6">
                        {toolsGrid.map((tool) => (
                            <div
                                key={tool.name}
                                onClick={() => handleIntegrationClick(tool.name)}
                                className="glass-card hover:bg-white/[0.08] flex flex-col items-center justify-center py-6 px-4 md:py-8 md:px-6 cursor-pointer transition-all duration-200 hover:-translate-y-1 hover:shadow-lg hover:shadow-accent/5 group"
                            >
                                <div className={`w-12 h-12 md:w-14 md:h-14 rounded-2xl flex items-center justify-center ${tool.bg} mb-3 transition-transform group-hover:scale-110`}>
                                    {tool.image ? (
                                        <img src={tool.image} alt={tool.name} className="w-6 h-6 md:w-7 md:h-7 object-contain" />
                                    ) : (
                                        <tool.icon size={24} className="md:w-7 md:h-7" style={{ color: tool.color }} />
                                    )}
                                </div>
                                <span className="text-xs text-white/50 group-hover:text-white/70 font-medium transition-colors">{tool.name}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </section>


            <div className="py-0 transform rotate-1 border-y-2 border-black relative z-20 mt-20 mb-10">
                <Marquee
                    items={['✓ SECURITY FIRST', '★ SOC2 COMPLIANT', '✓ ENTERPRISE READY', '★ 24/7 SUPPORT', '✓ CUSTOM MODELS']}
                    className="bg-[#FCD34D] text-black py-4"
                />
            </div>


            <footer className="relative z-10 border-t border-white/[0.06] bg-dark-800/50">
                <div className="max-w-6xl mx-auto px-6 py-12 md:py-16">
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-8 md:gap-6">


                        <div className="col-span-2 md:col-span-1">
                            <div className="mb-4">
                                <SetuLogo size="sm" />
                            </div>
                            <p className="text-xs text-white/30 leading-relaxed max-w-[200px]">
                                Build AI-agent lead users. Automate 5000+ custom use cases. All with AI.
                            </p>
                        </div>


                        {[
                            {
                                title: 'Product',
                                links: ['AI Agents', 'All Platforms', 'Virtual Assistant', 'Community'],
                            },
                            {
                                title: 'Company',
                                links: ['About Us', 'Contact', 'Affiliate Program', 'Onboard Community'],
                            },
                            {
                                title: 'Resources',
                                links: ['Learning Center', 'API Documentation', 'Security & Compliance', 'Shared Community'],
                            },
                            {
                                title: '',
                                links: ['', '', '', 'Send Feedback'],
                            },
                        ].map((col, i) => (
                            <div key={i}>
                                {col.title && (
                                    <h4 className="text-xs tracking-wider uppercase text-white/40 font-semibold mb-4">
                                        {col.title}
                                    </h4>
                                )}
                                <ul className="space-y-2.5">
                                    {col.links.filter(Boolean).map((link) => (
                                        <li key={link}>
                                            <a
                                                href="#"
                                                className="text-sm text-white/40 hover:text-white/70 transition-colors duration-200 flex items-center gap-1"
                                            >
                                                {link}
                                                {link === 'API Documentation' && <ExternalLink size={10} />}
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>

                    <div className="flex flex-col md:flex-row items-center justify-between mt-12 pt-8 border-t border-white/[0.05] gap-4">
                        <p className="text-xs text-white/25">
                            © 2026 Setu, Inc. All rights reserved.
                        </p>
                        <div className="flex items-center gap-6">
                            <a href="#" className="text-xs text-white/30 hover:text-white/50 transition-colors">Privacy Policy</a>
                            <a href="#" className="text-xs text-white/30 hover:text-white/50 transition-colors">Terms of Service</a>
                        </div>
                    </div>
                </div>
            </footer>


            {isAppsModalOpen && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6 md:p-12">

                    <div
                        className="absolute inset-0 bg-black/70 backdrop-blur-sm transition-opacity"
                        onClick={() => setIsAppsModalOpen(false)}
                    />


                    <div className="relative w-full max-w-5xl bg-[#141414] border border-white/10 rounded-2xl shadow-2xl flex flex-col max-h-[90vh] overflow-hidden">


                        <div className="flex items-center justify-between p-4 border-b border-white/5">
                            <div className="flex-1 max-w-md relative">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40" size={18} />
                                <input
                                    type="text"
                                    placeholder="Search apps by name or feature..."
                                    className="w-full bg-[#1A1A1A] border border-white/10 rounded-lg py-2 pl-10 pr-4 text-sm text-white focus:outline-none focus:border-white/30 focus:bg-[#202020] transition-colors"
                                />
                            </div>
                            <div className="flex items-center gap-4 ml-4">
                                <span className="text-sm text-white/50 hidden md:inline-block">Your agents can work with any of these apps</span>
                                <button
                                    onClick={() => setIsAppsModalOpen(false)}
                                    className="p-2 hover:bg-white/10 rounded-lg text-white/50 hover:text-white transition-colors"
                                >
                                    <X size={20} />
                                </button>
                            </div>
                        </div>


                        <div className="flex overflow-x-auto border-b border-white/5 px-2">
                            {tabs.map(tab => (
                                <button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    className={`flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${activeTab === tab
                                        ? 'border-accent text-white'
                                        : 'border-transparent text-white/40 hover:text-white/70'
                                        }`}
                                >
                                    {tab === 'Productivity' && <Layers size={16} />}
                                    {tab === 'Communication' && <MessageSquare size={16} />}
                                    {tab === 'Development' && <Code size={16} />}
                                    {tab === 'Business' && <Briefcase size={16} />}
                                    {tab === 'AI & More' && <Sparkles size={16} />}
                                    {tab}
                                </button>
                            ))}
                        </div>


                        <div className="flex-1 overflow-y-auto p-6 space-y-8">


                            {activeTab === 'Productivity' && appsData['Popular Apps'] && (
                                <div>
                                    <h3 className="flex items-center gap-2 text-sm font-semibold text-white/90 mb-4">
                                        <Star size={16} className="text-yellow-500" fill="currentColor" />
                                        Popular Apps
                                        <span className="text-xs text-white/40 font-normal ml-2">Most used by agents</span>
                                    </h3>
                                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                        {appsData['Popular Apps'].map(app => (
                                            <AppCard
                                                key={`popular-${app.name}`}
                                                app={app}
                                                onClick={() => {
                                                    setIsAppsModalOpen(false);
                                                    setTimeout(() => handleIntegrationClick(app.name), 100);
                                                }}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )}


                            <div>
                                <h3 className="text-sm font-semibold text-white/90 mb-4">{activeTab}</h3>
                                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                    {appsData[activeTab]?.map(app => (
                                        <AppCard
                                            key={`${activeTab}-${app.name}`}
                                            app={app}
                                            onClick={() => {
                                                setIsAppsModalOpen(false);
                                                setTimeout(() => handleIntegrationClick(app.name), 100);
                                            }}
                                        />
                                    ))}
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            )}
        </div>
    );
}
