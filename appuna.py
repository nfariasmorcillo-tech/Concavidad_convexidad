import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuración de la página web
st.set_page_config(page_title="Análisis de Monotonía y Concavidad", page_icon="📉", layout="wide")

st.title("📉 Estudio Completo de Funciones: Monotonía y Concavidad ($f(x)$)")
st.markdown("Herramienta pedagógica basada en algoritmos analíticos para identificar intervalos de crecimiento, decrecimiento, concavidad y puntos críticos.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Configuración de la Función")

func_str = st.sidebar.text_input("Función f(x):", value="x**3 - 3*x")
var_str = st.sidebar.text_input("Variable autónoma:", value="x")

col_range1, col_range2 = st.sidebar.columns(2)
xmin = col_range1.number_input("x mínimo (xmin):", value=-3.0)
xmax = col_range2.number_input("x máximo (xmax):", value=3.0)

st.sidebar.divider()
st.sidebar.info(
    "**Código de colores en gráficos:**\n\n"
    "**Gráfico 1: Monotonía**\n"
    "* 🔵 Azul: Creciente ($f'(x) > 0$)\n"
    "* 🔴 Rojo: Decreciente ($f'(x) < 0$)\n\n"
    "**Gráfico 2: Concavidad**\n"
    "* 🟢 Verde: Cóncava hacia arriba ($f''(x) > 0$)\n"
    "* 🟠 Naranja: Cóncava hacia abajo ($f''(x) < 0$)"
)

# Botón principal para ejecutar el análisis
analizar = st.sidebar.button("Analizar Función", type="primary")

# =========================================================================
# LÓGICA ALGEBRAICA (Réplica analítica de los algoritmos de Mathematica)
# =========================================================================
if analizar:
    try:
        # Definir símbolos y procesar la función string
        x = sp.Symbol(var_str.strip())
        func = sp.sympify(func_str)
        
        # --- CONTROL DE EXCEPCIONES TRIGONOMÉTRICAS ---
        if any(func.has(trig) for trig in [sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc]):
            st.error("❌ Error: La función contiene términos trigonométricos restringidos para este análisis de intervalos.")
        else:
            # --- 1. CÁLCULO DE DERIVADAS ---
            fp1 = sp.diff(func, x)  # Primera derivada (Monotonía)
            fp2 = sp.diff(fp1, x)   # Segunda derivada (Concavidad)
            
            st.header("1. Estudio Analítico de Derivadas")
            c_der1, c_der2 = st.columns(2)
            c_der1.latex(rf"f'(x) = \frac{{df}}{{dx}} = {sp.latex(fp1)}")
            c_der2.latex(rf"f''(x) = \frac{{d^2f}}{{dx^2}} = {sp.latex(fp2)}")
            
            st.divider()

            # Determinar puntos de discontinuidad de la función original (cpdf)
            _, den_original = sp.fraction(sp.together(func))
            cpdf = [float(r.evalf()) for r in sp.solve(sp.Eq(den_original, 0), x) if r.is_real]

            # =========================================================================
            # PROCESAMIENTO DE MONOTONÍA (Crecimiento / Decrecimiento)
            # =========================================================================
            to_fp1 = sp.together(fp1)
            num_m, den_m = sp.fraction(to_fp1)
            
            cpn_m = [float(r.evalf()) for r in sp.solve(sp.Eq(num_m, 0), x) if r.is_real]
            cpd_m = [float(r.evalf()) for r in sp.solve(sp.Eq(den_m, 0), x) if r.is_real]
            
            pc_monotonia = sorted(list(set(cpn_m + cpd_m)))
            pc_m_rango = [p for p in pc_monotonia if xmin <= p <= xmax]
            puntos_extremos_validos = [p for p in pc_m_rango if p not in cpdf]

            fronteras_m = [xmin] + pc_m_rango + [xmax]
            intervalos_m = []

            for i in range(len(fronteras_m) - 1):
                start, end = fronteras_m[i], fronteras_m[i+1]
                pmedio = (start + end) / 2.0
                valor_f1 = float(fp1.subs(x, pmedio).evalf())
                
                if valor_f1 > 0:
                    estado = "🔵 Creciente (↗️)"
                    color = "#2980b9"  # Azul
                elif valor_f1 < 0:
                    estado = "🔴 Decreciente (↘️)"
                    color = "#c0392b"  # Rojo
                else:
                    estado = "⚪ Constante / Estacionario"
                    color = "#7f8c8d"
                    
                intervalos_m.append({
                    "Intervalo": f"({start:.2f} , {end:.2f})", "Punto Prueba": round(pmedio, 2),
                    "f'(x) Evaluada": f"{valor_f1:.4f}", "Estado": estado, "Color": color, "Inicio": start, "Fin": end
                })

            # =========================================================================
            # PROCESAMIENTO DE CONCAVIDAD
            # =========================================================================
            to_fp2 = sp.together(fp2)
            num_c, den_c = sp.fraction(to_fp2)
            
            cpn_c = [float(r.evalf()) for r in sp.solve(sp.Eq(num_c, 0), x) if r.is_real]
            cpd_c = [float(r.evalf()) for r in sp.solve(sp.Eq(den_c, 0), x) if r.is_real]
            
            pc_concavidad = sorted(list(set(cpn_c + cpd_c)))
            pc_c_rango = [p for p in pc_concavidad if xmin <= p <= xmax]
            puntos_inflexion_validos = [p for p in pc_c_rango if p not in cpdf]

            fronteras_c = [xmin] + pc_c_rango + [xmax]
            intervalos_c = []

            for i in range(len(fronteras_c) - 1):
                start, end = fronteras_c[i], fronteras_c[i+1]
                pmedio = (start + end) / 2.0
                valor_f2 = float(fp2.subs(x, pmedio).evalf())
                
                if valor_f2 > 0:
                    estado = "🟢 Cóncava hacia arriba (∪)"
                    color = "#2ecc71"  # Verde
                elif valor_f2 < 0:
                    estado = "🟠 Cóncava hacia abajo (∩)"
                    color = "#e67e22"  # Naranja
                else:
                    estado = "⚪ Inflexión / Lineal"
                    color = "#95a5a6"
                    
                intervalos_c.append({
                    "Intervalo": f"({start:.2f} , {end:.2f})", "Punto Prueba": round(pmedio, 2),
                    "f''(x) Evaluada": f"{valor_f2:.4f}", "Estado": estado, "Color": color, "Inicio": start, "Fin": end
                })

            # =========================================================================
            # RENDERIZADO DE TABLAS EN PESTAÑAS (Estilo Grid de Mathematica)
            # =========================================================================
            st.header("2. Tablas Analíticas de Intervalos")
            tab1, tab2 = st.tabs(["📊 Monotonía (f'(x))", "📊 Concavidad (f''(x))"])
            
            with tab1:
                df_m = pd.DataFrame(intervalos_m)
                st.dataframe(df_m[["Intervalo", "Punto Prueba", "f'(x) Evaluada", "Estado"]], use_container_width=True, hide_index=True)
                if pc_monotonia:
                    st.latex(rf"\text{{Puntos críticos de primer orden encontrados: }} x \in {sp.latex(pc_monotonia)}")
            
            with tab2:
                df_c = pd.DataFrame(intervalos_c)
                st.dataframe(df_c[["Intervalo", "Punto Prueba", "f''(x) Evaluada", "Estado"]], use_container_width=True, hide_index=True)
                if pc_concavidad:
                    st.latex(rf"\text{{Puntos críticos de segundo orden encontrados: }} x \in {sp.latex(pc_concavidad)}")

            st.divider()

            # =========================================================================
            # VISUALIZACIÓN GRÁFICA INTERACTIVA TRÁMÓ A TRAMO
            # =========================================================================
            st.header("3. Representación Gráfica del Comportamiento")
            col_g1, col_g2 = st.columns(2)

            # --- GRÁFICO 1: MONOTONÍA ---
            with col_g1:
                st.subheader("Gráfico de Crecimiento y Decrecimiento")
                fig1 = go.Figure()
                for inter in intervalos_m:
                    x_vals = np.linspace(inter["Inicio"], inter["Fin"], 100)
                    y_vals = [float(func.subs(x, val).evalf()) if val not in cpdf else None for val in x_vals]
                    fig1.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='lines',
                        line=dict(color=inter["Color"], width=4),
                        name=inter["Estado"].split()[-1], hoverinfo='x+y'
                    ))
                # Marcar puntos extremos locales
                if puntos_extremos_validos:
                    pe_y = [float(func.subs(x, pt).evalf()) for pt in puntos_extremos_validos]
                    fig1.add_trace(go.Scatter(
                        x=puntos_extremos_validos, y=pe_y, mode='markers',
                        marker=dict(color='black', size=10, line=dict(color='white', width=2)),
                        name='Extremo local', text=[f"Punto Crítico<br>x: {pt:.2f}<br>y: {func.subs(x, pt).evalf():.2f}" for pt in puntos_extremos_validos],
                        hoverinfo='text'
                    ))
                fig1.update_layout(xaxis_title=f"Eje {var_str}", yaxis_title="f(x)", showlegend=False, template="plotly_white", height=450)
                fig1.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='Black', gridcolor='LightGray')
                fig1.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='Black', gridcolor='LightGray')
                st.plotly_chart(fig1, use_container_width=True)

            # --- GRÁFICO 2: CONCAVIDAD ---
            with col_g2:
                st.subheader("Gráfico de Concavidad e Inflexión")
                fig2 = go.Figure()
                for inter in intervalos_c:
                    x_vals = np.linspace(inter["Inicio"], inter["Fin"], 100)
                    y_vals = [float(func.subs(x, val).evalf()) if val not in cpdf else None for val in x_vals]
                    fig2.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='lines',
                        line=dict(color=inter["Color"], width=4),
                        name=inter["Estado"].split()[-1], hoverinfo='x+y'
                    ))
                # Marcar puntos de inflexión geométricos
                if puntos_inflexion_validos:
                    pi_y = [float(func.subs(x, pt).evalf()) for pt in puntos_inflexion_validos]
                    fig2.add_trace(go.Scatter(
                        x=puntos_inflexion_validos, y=pi_y, mode='markers',
                        marker=dict(color='black', size=10, line=dict(color='white', width=2)),
                        name='Inflexión', text=[f"Inflexión<br>x: {pt:.2f}<br>y: {func.subs(x, pt).evalf():.2f}" for pt in puntos_inflexion_validos],
                        hoverinfo='text'
                    ))
                fig2.update_layout(xaxis_title=f"Eje {var_str}", yaxis_title="f(x)", showlegend=False, template="plotly_white", height=450)
                fig2.update_xaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='Black', gridcolor='LightGray')
                fig2.update_yaxes(zeroline=True, zerolinewidth=1.5, zerolinecolor='Black', gridcolor='LightGray')
                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"Error analítico: {e}. Revisa la sintaxis ingresada (ejemplo: `(x**2 - 1) / (x - 2)`).")
