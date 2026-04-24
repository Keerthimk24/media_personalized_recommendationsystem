"""
Step 5: Streamlit Dashboard
----------------------------
Netflix-style UI for the hybrid recommendation engine.
Users log in, see their history, and get personalized recommendations.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import streamlit as st
import pandas as pd

# ─── Page config (MUST be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="MediaStream — AI Recommendations",
    layout="wide",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# ─── Premium CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e2e8f0; }

  .stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 30%, #0a0a1a 60%, #111133 100%);
    background-attachment: fixed;
  }

  /* Animated background orbs */
  .bg-orbs {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
  }
  .orb {
    position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.15;
    animation: float 20s ease-in-out infinite;
  }
  .orb1 { width: 600px; height: 600px; background: #e50914; top: -10%; left: -5%; animation-delay: 0s; }
  .orb2 { width: 500px; height: 500px; background: #6366f1; bottom: -10%; right: -5%; animation-delay: -7s; }
  .orb3 { width: 400px; height: 400px; background: #f59e0b; top: 50%; left: 50%; animation-delay: -14s; }
  @keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(30px, -30px) scale(1.05); }
    66% { transform: translate(-20px, 20px) scale(0.95); }
  }

  /* Hero */
  .hero-container { text-align: center; padding: 2rem 0 1rem; position: relative; }
  .hero-title {
    font-size: 3.8rem; font-weight: 900; letter-spacing: -3px; margin-bottom: 0;
    background: linear-gradient(135deg, #e50914 0%, #ff6b6b 40%, #ffa07a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-fill-color: transparent;
    animation: glow 3s ease-in-out infinite alternate;
  }
  @keyframes glow {
    from { filter: drop-shadow(0 0 20px rgba(229,9,20,0.3)); }
    to { filter: drop-shadow(0 0 40px rgba(229,9,20,0.5)); }
  }
  .hero-badge {
    display: inline-block; margin-top: 0.8rem;
    background: linear-gradient(135deg, rgba(229,9,20,0.15), rgba(99,102,241,0.15));
    border: 1px solid rgba(229,9,20,0.3); border-radius: 50px;
    padding: 8px 24px; font-size: 0.85rem; color: #f8fafc;
    letter-spacing: 2px; font-weight: 500; text-transform: uppercase;
    backdrop-filter: blur(10px);
  }
  .hero-sub {
    text-align: center; color: #64748b; font-size: 1rem;
    margin: 0.5rem 0 2rem; font-weight: 400; letter-spacing: 0.5px;
  }

  /* Movie card */
  .movie-card {
    background: rgba(15, 20, 40, 0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 0; margin-bottom: 1.2rem;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(20px); overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  .movie-card:hover {
    border-color: rgba(229,9,20,0.6);
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 50px rgba(229,9,20,0.2), 0 0 0 1px rgba(229,9,20,0.3);
  }
  .card-poster {
    width: 100%; aspect-ratio: 16/9;
    display: flex; align-items: center; justify-content: center;
    font-size: 3rem; position: relative;
    border-bottom: 1px solid rgba(255,255,255,0.05);
  }
  .card-body { padding: 1rem 1rem 0.8rem; }
  .movie-title {
    font-size: 0.95rem; font-weight: 700; color: #fff;
    margin: 0 0 0.4rem 0; line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .movie-year { font-size: 0.75rem; color: #94a3b8; font-weight: 600; }

  /* Genre badge */
  .genre-badge {
    display: inline-block;
    background: rgba(229,9,20,0.12); border: 1px solid rgba(229,9,20,0.25);
    border-radius: 6px; padding: 2px 8px;
    font-size: 0.65rem; font-weight: 600; color: #ff6b6b;
    margin: 2px 2px 4px 0;
  }

  /* Language tag */
  .lang-tag {
    display: inline-block;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4);
    border-radius: 6px; padding: 2px 8px;
    font-size: 0.65rem; font-weight: 700; color: #a5b4fc;
  }

  /* Score bar */
  .score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 10px; height: 5px; margin: 8px 0; overflow: hidden;
  }
  .score-bar-fill {
    background: linear-gradient(90deg, #e50914, #ff6b6b, #ffa07a);
    height: 100%; border-radius: 10px;
    animation: fillBar 1.2s ease-out;
  }
  @keyframes fillBar { from { width: 0% !important; } }

  .movie-meta { font-size: 0.75rem; color: #64748b; margin-bottom: 0.3rem; }

  /* Reason box */
  .reason-box {
    background: linear-gradient(135deg, rgba(229,9,20,0.08), rgba(99,102,241,0.08));
    border-left: 3px solid #e50914; border-radius: 0 8px 8px 0;
    padding: 6px 10px; font-size: 0.7rem;
    color: #cbd5e1; line-height: 1.4; margin-top: 0.5rem;
  }

  /* History card */
  .history-card {
    background: rgba(15,20,40,0.5); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;
    font-size: 0.85rem; transition: all 0.3s ease;
    display: flex; align-items: center; gap: 12px;
  }
  .history-card:hover { border-color: rgba(229,9,20,0.3); background: rgba(15,20,40,0.8); }

  /* Section header */
  .section-header {
    font-size: 1.6rem; font-weight: 800; color: #fff;
    padding-bottom: 0.5rem; margin: 2rem 0 1.2rem 0;
    position: relative; display: inline-block;
  }
  .section-header::after {
    content: ''; position: absolute; bottom: 0; left: 0;
    width: 60px; height: 3px; border-radius: 3px;
    background: linear-gradient(90deg, #e50914, #ff6b6b);
  }

  /* Metric card */
  .metric-card {
    background: rgba(15,20,40,0.6); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.2rem; text-align: center;
    backdrop-filter: blur(20px); transition: all 0.3s ease;
  }
  .metric-card:hover { transform: translateY(-4px); border-color: rgba(229,9,20,0.3); }
  .metric-value { font-size: 1.8rem; font-weight: 900; color: #e50914; margin-bottom: 4px; }
  .metric-label {
    font-size: 0.7rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 2px; font-weight: 600;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,8,24,0.98), rgba(15,15,40,0.98)) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
  }
  [data-testid="stSidebar"] .block-container { padding-top: 2rem; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #e50914, #c20710) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.7rem 2rem !important;
    transition: all 0.3s ease !important; letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(229,9,20,0.3) !important;
  }
  .stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 10px 30px rgba(229,9,20,0.5) !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: rgba(15,20,40,0.5) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; font-weight: 600 !important;
  }

  /* Divider */
  .custom-divider {
    height: 1px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, rgba(229,9,20,0.3), transparent);
  }

  /* Stats row */
  .stats-row {
    display: flex; gap: 12px; margin: 1rem 0;
    flex-wrap: wrap;
  }
  .stat-chip {
    background: rgba(15,20,40,0.6); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 6px 14px; font-size: 0.75rem;
    color: #94a3b8; backdrop-filter: blur(10px);
  }
  .stat-chip b { color: #e2e8f0; }

  /* Score display */
  .score-display {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.8rem; font-weight: 700; color: #ffa07a;
  }

  /* Hide Streamlit branding */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Background orbs
st.markdown("""
<div class="bg-orbs">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>
""", unsafe_allow_html=True)


# ─── Load Engine (cached) ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    try:
        from step4_engine import RecommendationEngine
        return RecommendationEngine()
    except Exception as e:
        st.error(f"❌ Models not found or error loading engine! Error: {e}")
        st.code("python step1_preprocess.py\npython step3_train.py")
        st.stop()


# ─── Helpers ──────────────────────────────────────────────────────────────
def build_reason(rec: dict) -> str:
    """Build a human-readable explanation for a recommendation."""
    parts = []
    if rec.get('ncf_score', 0) > 0.5:
        parts.append("Users like you loved this")
    if rec.get('content_score', 0) > 0.3:
        parts.append("Matches your genre taste")
    if rec.get('lang_boost', 0) > 0.1:
        lang_name = rec.get('language', rec.get('primary_language', ''))
        parts.append(f"In your preferred language ({lang_name})")
    if not parts:
        parts.append("Trending & highly rated")
    return " | ".join(parts)


def render_movie_card(rec: dict, col):
    """Render a single movie recommendation card."""
    with col:
        genre_icons = {
            'Action': '💥', 'Comedy': '😂', 'Drama': '🎭',
            'Horror': '👻', 'Romance': '💕', 'Sci-Fi': '🚀',
            'Science Fiction': '🚀', 'Thriller': '🔪',
            'Animation': '🎨', 'Family': '👨‍👩‍👧', 'Fantasy': '🧙',
            'Adventure': '🗺️', 'Mystery': '🔍', 'Crime': '🔫',
            'Documentary': '📹', 'Music': '🎵', 'War': '⚔️',
        }
        genre_colors = {
            'Action': '#e50914', 'Comedy': '#f59e0b', 'Drama': '#6366f1',
            'Horror': '#7c3aed', 'Romance': '#ec4899', 'Sci-Fi': '#06b6d4',
            'Science Fiction': '#06b6d4', 'Thriller': '#dc2626',
            'Animation': '#10b981', 'Family': '#f97316', 'Fantasy': '#8b5cf6',
            'Adventure': '#14b8a6', 'Mystery': '#6366f1', 'Crime': '#ef4444',
        }
        genres_raw = rec.get('genres_str', rec.get('genres', ''))
        genres = str(genres_raw).split('|')
        icon = '🎬'
        accent = '#e50914'
        for g in genres:
            g = g.strip()
            if g in genre_icons:
                icon = genre_icons[g]
                accent = genre_colors.get(g, '#e50914')
                break

        score_pct = int(min(rec.get('hybrid_score', 0) * 100, 100))
        reason = build_reason(rec)
        genres_list = str(genres_raw).split('|')[:3]
        genre_display = ''.join([f'<span class="genre-badge">{g.strip()}</span>' for g in genres_list if g.strip()])

        year_display = rec.get('year', '')
        if year_display is None or pd.isna(year_display) or year_display == "":
            year_display = "Classic"
        else:
            try:
                year_display = str(int(float(year_display)))
            except:
                pass

        title = rec.get('title', 'Unknown Movie')
        lang = rec.get('language', rec.get('primary_language', 'Unknown'))
        match_score = rec.get('hybrid_score', 0)

        st.markdown(f"""
        <div class="movie-card">
          <div class="card-poster" style="background: linear-gradient(135deg, {accent}22 0%, #0a0a1a 100%);">
            <span style="font-size:2.5rem;">{icon}</span>
          </div>
          <div class="card-body">
            <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:4px;">
              <div class="movie-title" style="flex:1;">{title}</div>
            </div>
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:6px;">
              <span class="movie-year">{year_display}</span>
              <span class="lang-tag">{lang}</span>
            </div>
            <div>{genre_display}</div>
            <div class="score-bar-bg">
              <div class="score-bar-fill" style="width:{score_pct}%"></div>
            </div>
            <div class="movie-meta">Match: <b style="color:#ffa07a">{round(match_score, 2)}</b></div>
            <div class="reason-box">💡 {reason}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─── MAIN APP ─────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero-container">
      <div class="hero-title">MediaStream</div>
      <div class="hero-badge">🧠 Powered by Deep Learning</div>
      <div class="hero-sub">NCF + TF-IDF Content Similarity · Multi-Language · Explainable AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Load engine
    with st.spinner("Loading AI models..."):
        engine = load_engine()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
          <div style="font-size:1.8rem;font-weight:900;
            background:linear-gradient(135deg,#e50914,#ff6b6b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;">🎬 MediaStream</div>
          <div style="font-size:0.7rem;color:#475569;letter-spacing:2px;
            text-transform:uppercase;margin-top:4px;">AI Engine v2.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 👤 User Profile")

        # Build user list from survey
        if hasattr(engine, "survey") and not engine.survey.empty:
            survey_users = sorted(
                engine.survey['user_id'].unique().tolist(),
                key=lambda x: int(x[1:]) if str(x)[1:].isdigit() else 0
            )
        else:
            survey_users = []

        users = ["New User (Cold Start)"] + survey_users
        selected_user = st.selectbox("Select User", users, index=0)

        num_recs = st.slider("Number of Recommendations", 5, 20, 10)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 📊 System Stats")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{engine.num_items:,}</div>
              <div class="metric-label">Movies</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{engine.num_users}</div>
              <div class="metric-label">Users</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 🧮 Hybrid Formula")
        st.markdown("""
        <div style="font-size:0.75rem;color:#64748b;line-height:1.8;">
          <div class="stat-chip" style="margin-bottom:6px;display:block;">
            <span style="color:#e50914;font-weight:700;">0.45×</span> NCF <span style="color:#475569;">(Deep Learning)</span>
          </div>
          <div class="stat-chip" style="margin-bottom:6px;display:block;">
            <span style="color:#6366f1;font-weight:700;">0.30×</span> Content <span style="color:#475569;">(TF-IDF)</span>
          </div>
          <div class="stat-chip" style="margin-bottom:6px;display:block;">
            <span style="color:#10b981;font-weight:700;">0.25×</span> Language Boost
          </div>
        </div>
        <div style="margin-top:10px;font-size:0.7rem;color:#475569;">
          🌐 <b style="color:#94a3b8;">Languages:</b> English, Hindi, Telugu
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ──────────────────────────────────────────────────────
    if selected_user == "New User (Cold Start)":
        _cold_start_ui(engine, num_recs)
    else:
        _returning_user_ui(engine, selected_user, num_recs)


# ─── Cold Start ──────────────────────────────────────────────────────────
def _cold_start_ui(engine, num_recs):
    st.markdown('<div class="section-header">✨ New here? Tell us what you love</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem;color:#64748b;margin-bottom:1.5rem;">
      Pick your favorite genres and language — our AI will find your perfect matches.
    </div>
    """, unsafe_allow_html=True)

    all_genres = engine.get_all_genres()
    all_langs = engine.get_all_languages()

    c1, c2 = st.columns(2)
    with c1:
        pref_genres = st.multiselect("🎭 Select Genres", all_genres,
                                     default=["Action"],
                                     help="Only movies matching these genres will be shown")
    with c2:
        pref_lang = st.selectbox("🌐 Preferred Language",
                                 ["Any"] + all_langs, index=0)

    if "show_cold_recs" not in st.session_state:
        st.session_state.show_cold_recs = False

    if st.button("🎬 Discover Movies", use_container_width=True):
        st.session_state.show_cold_recs = True

    if st.session_state.show_cold_recs:
        with st.spinner("🔍 Finding your perfect matches..."):
            recs = engine._cold_start_recommend("COLD", num_recs,
                                                genre_filter=pref_genres if pref_genres else None,
                                                lang_filter=pref_lang)
            recs = recs[:num_recs]

        if not recs:
            st.warning("No movies found for this genre/language. Try different filters.")
            return

        genre_label = ', '.join(pref_genres) if pref_genres else 'All Genres'
        lang_label = f' · {pref_lang}' if pref_lang != 'Any' else ''
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">🎯 {genre_label}{lang_label} — Top Picks</div>',
                    unsafe_allow_html=True)
        _render_grid(recs)


# ─── Returning User ──────────────────────────────────────────────────────
def _returning_user_ui(engine, user_id, num_recs):
    st.markdown(f'<div class="section-header">👋 Welcome back, {user_id}</div>',
                unsafe_allow_html=True)

    # ── User Preferences Summary ──
    prefs = engine.get_user_preferences(user_id)
    history = engine.get_user_history(user_id)

    if prefs.get('genres') or prefs.get('languages'):
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            genres_txt = ", ".join(prefs.get('genres', [])[:3]) or "Exploring"
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.5rem;">🎭</div>
              <div class="metric-value" style="font-size:0.9rem;color:#ff6b6b">{genres_txt}</div>
              <div class="metric-label">Fav Genres</div>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            langs_txt = ", ".join(prefs.get('languages', [])[:3]) or "All"
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.5rem;">🌐</div>
              <div class="metric-value" style="font-size:0.9rem;color:#818cf8">{langs_txt}</div>
              <div class="metric-label">Languages</div>
            </div>
            """, unsafe_allow_html=True)
        with p3:
            avg_r = prefs.get('avg_rating', 0)
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.5rem;">⭐</div>
              <div class="metric-value" style="font-size:0.9rem;color:#f59e0b">{round(avg_r, 1)}/5.0</div>
              <div class="metric-label">Avg Rating</div>
            </div>
            """, unsafe_allow_html=True)
        with p4:
            watch_count = len(history) if history is not None and not history.empty else 0
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.5rem;">📺</div>
              <div class="metric-value" style="font-size:0.9rem;color:#10b981">{watch_count}</div>
              <div class="metric-label">Watched</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Survey Info ──
    survey_row = engine.survey[engine.survey['user_id'] == user_id]
    if not survey_row.empty:
        row = survey_row.iloc[0]
        with st.expander("📋 Your Survey Profile", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.write(f"**Preferred Genres:** {row.get('preferred_genres', 'N/A')}")
                st.write(f"**Content Type:** {row.get('content_type', 'N/A')}")
                st.write(f"**Watch Time:** {row.get('watch_time', 'N/A')}")
            with sc2:
                st.write(f"**Device:** {row.get('device', 'N/A')}")
                st.write(f"**Rewatch Frequency:** {row.get('rewatch_freq', 'N/A')}")
                st.write(f"**Recent Favorite:** {row.get('recent_favorite', 'N/A')}")

    # ── Watch History ──
    if history is not None and not history.empty:
        with st.expander(f"🕐 Watch History ({len(history)} movies)", expanded=False):
            for _, h in history.head(10).iterrows():
                title = h.get('title', 'Unknown')
                lang = h.get('primary_language', '')
                genres = str(h.get('genres_str', '')).replace('|', ' · ')[:40]
                rating = h.get('rating', 0)
                stars = '⭐' * int(rating)
                st.markdown(f"""
                <div class="history-card">
                  <div style="flex:1;">
                    <b style="color:#fff;">{title}</b>
                    <span class="lang-tag" style="margin-left:8px;">{lang}</span>
                    <div style="font-size:0.7rem;color:#475569;margin-top:4px;">{genres}</div>
                  </div>
                  <div style="text-align:right;min-width:80px;">
                    <span style="font-size:0.85rem;">{stars}</span>
                    <div style="font-size:0.7rem;color:#94a3b8;">{round(rating, 1)}/5</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Genre & Language Filter ──
    all_genres = engine.get_all_genres()
    all_langs = engine.get_all_languages()
    fc1, fc2 = st.columns(2)
    with fc1:
        genre_filter = st.multiselect("Filter by Genre", all_genres,
                                      help="Select a genre to see ONLY that type of movie")
    with fc2:
        lang_filter = st.selectbox("Filter by Language",
                                   ["Any"] + all_langs, index=0,
                                   help="Select a language to see ONLY that language")

    # ── Get Recommendations ──
    if "show_recs" not in st.session_state:
        st.session_state.show_recs = False

    if st.button("🚀 Get AI Recommendations", use_container_width=True):
        st.session_state.show_recs = True

    if st.session_state.show_recs:
        with st.spinner("🧠 AI is analyzing your taste..."):
            recs = engine.recommend(user_id, top_k=num_recs,
                                    genre_filter=genre_filter if genre_filter else None,
                                    lang_filter=lang_filter if lang_filter != 'Any' else None)

        if not recs:
            st.info("No movies found for this genre. Try a different filter.")
            return

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎯 Your Personalized Recommendations</div>',
                    unsafe_allow_html=True)

        avg_ncf = sum(r['ncf_score'] for r in recs) / len(recs)
        avg_content = sum(r['content_score'] for r in recs) / len(recs)
        avg_lang = sum(r['lang_boost'] for r in recs) / len(recs)

        mc1, mc2, mc3, mc4 = st.columns(4)
        for mc, lbl, val, color, icon in [
            (mc1, "Results", str(len(recs)), "#e50914", "🎬"),
            (mc2, "NCF Score", f"{round(avg_ncf, 3)}", "#f59e0b", "🧠"),
            (mc3, "Content", f"{round(avg_content, 3)}", "#6366f1", "📊"),
            (mc4, "Lang Boost", f"{round(avg_lang, 3)}", "#10b981", "🌐"),
        ]:
            with mc:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:1.2rem;">{icon}</div>
                  <div class="metric-value" style="color:{color}">{val}</div>
                  <div class="metric-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        _render_grid(recs)


# ─── Render Grid ─────────────────────────────────────────────────────────
def _render_grid(recs):
    N_COLS = 4
    for row_start in range(0, len(recs), N_COLS):
        row_recs = recs[row_start:row_start + N_COLS]
        cols = st.columns(N_COLS)
        for i, rec in enumerate(row_recs):
            render_movie_card(rec, cols[i])


if __name__ == "__main__":
    main()
