import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuración de la página web
st.set_page_config(page_title="Análisis de Concavidad e Intervalos", page_icon="📉", layout="wide")

st.title("📉 Análisis de Concavidad e Intervalos Gráficos ($f(x)$)")
st.markdown("Herramienta pedagógica basada en el algoritmo `ConcavityIntervalsPlot` para identificar analíticamente intervalos de concavidad y puntos de inflexión.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Configuración de la Función")

func_str = st.sidebar.text_input("Función f(x):", value="x**4 - 6*x**2 + 5")
var_str = st.sidebar.text_input("Variable autónoma:", value="x")

col_range1, col_range2 = st.sidebar.columns(2)
xmin = col_range1.number_input("x mínimo (xmin):", value=-3.0)
xmax = col_range2.number_input("x máximo (xmax):", value=3.0)

st.sidebar.divider()
st.sidebar.info(
    "**Código de color pedagógico:**\n"
    "* 🟢 Verde: Concavidad hacia arriba ($f''(x) > 0$).\n"
    "* 🟠 Naranja: Concavidad hacia abajo ($f''(x) < 0$)."
)

# Botón principal para ejecutar el análisis
analizar = st.sidebar.button("Analizar Concavidad", type="primary")

# =========================================================================
# LÓGICA ALGEBRAICA (Réplica exacta de ConcavityIntervalsPlot)
# =========================================================================
if analizar:
    try:
        # Definir símbolos y procesar la función string
        x = sp.Symbol(var_str.strip())
        func = sp.sympify(func_str)
        
        # --- CONTROL DE EXCEPCIONES TRIGONOMÉTRICAS ---
        # Equivale a: If[! FreeQ[func, Sin | Cos | ...], Return[Message[...]]]
        if any(func.has(trig) for trig in [sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc]):
            st.error("❌ Error: La función contiene términos trigonométricos no lineales excluidos para este análisis de intervalos.")
        else:
            # --- 1. CÁLCULO DE DERIVADAS ---
            fp1 = sp.diff(func, x)
            fp2 = sp.diff(fp1, x)   # Segunda derivada (fp en Mathematica)
            
            st.header("1. Estudio Analítico de Derivadas")
            c_der1, c_der2 = st.columns(2)
            c_der1.latex(rf"f'(x) = \frac{{df}}{{dx}} = {sp.latex(fp1)}")
            c_der2.latex(rf"f''(x) = \frac{{d^2f}}{{dx^2}} = {sp.latex(fp2)}")
            
            st.divider()

            # --- 2. EXTRACCIÓN DE RAÍCES SIN PÉRDIDA ALGEBRAICA ---
            # Equivale a: tofp = Together[fp]; {num, den} = {Numerator[tofp], Denominator[tofp]}
            tofp = sp.together(fp2)
            num, den = sp.fraction(tofp)
            
            # Hallar los ceros reales del numerador (cpn) y denominador (cpd)
            raices_num = sp.solve(sp.Eq(num, 0), x)
            raices_den = sp.solve(sp.Eq(den, 0), x)
            
            # Filtro estricto de valores reales numéricos
            cpn = [float(r.evalf()) for r in raices_num if r.is_real]
            cpd = [float(r.evalf()) for r in raices_den if r.is_real]
            
            # Conjunto unión ordenado de puntos críticos totales (pc = Sort[Union[cpn, cpd]])
            puntos_criticos = sorted(list(set(cpn + cpd)))
            
            # Filtrar solo aquellos que caen en la ventana de visualización del usuario
            puntos_en_rango = [p for p in puntos_criticos if xmin <= p <= xmax]
            
            # Determinar puntos de discontinuidad de la función original (cpdf)
            _, den_original = sp.fraction(sp.together(func))
            raices_cpdf = sp.solve(sp.Eq(den_original, 0), x)
            cpdf = [float(r.evalf()) for r in raices_cpdf if r.is_real]
            
            # Los puntos válidos para graficar discos son el complemento de los críticos con las discontinuidades
            puntos_inflexion_validos = [p for p in puntos_en_rango if p not in cpdf]

            st.header("2. Puntos Críticos y Puntos de Inflexión")
            if puntos_criticos:
                st.latex(rf"\text{{Puntos críticos de concavidad encontrados (}} f''(x) = 0 \text{{ o No Existe): }} x \in {sp.latex(puntos_criticos)}")
            else:
                st.markdown("No se encontraron puntos críticos reales que cambien la concavidad.")

            st.divider()

            # --- 3. CONSTRUCCIÓN DE INTERVALOS Y ANÁLISIS DE SIGNO ---
            # Equivale a estructurar Partition[Append[Prepend[pc, xmin], xmax], {2}, {1}]
            puntos_frontera = [xmin] + puntos_en_rango + [xmax]
            intervalos = []
            
            st.header("3. Tabla Analítica de Signos por Intervalos")
            
            for i in range(len(puntos_frontera) - 1):
                start, end = puntos_frontera[i], puntos_frontera[i+1]
                pmedio = (start + end) / 2.0  # Punto de prueba intermedio
                
                # Evaluar numéricamente el signo de la segunda derivada en el punto de prueba
                valor_f2 = float(fp2.subs(x, pmedio).evalf())
                
                if valor_f2 > 0:
                    estado = "🟢 Concavidad Hacia Arriba (Cóncava)"
                    color_trama = "#2ecc71"  # Verde plano elegante
                elif valor_f2 < 0:
                    estado = "🟠 Concavidad Hacia Abajo (Convexa)"
                    color_trama = "#e67e22"  # Naranja plano elegante
                else:
                    estado = "⚪ Punto de Inflexión / Lineal"
                    color_trama = "#95a5a6"
                    
                intervalos.append({
                    "Intervalo": f"({start:.2f} , {end:.2f})",
                    "Punto de Prueba": round(pmedio, 2),
                    "f''(x) Evaluada": f"{valor_f2:.4f}",
                    "Estado Analítico": estado,
                    "Color": color_trama,
                    "Inicio": start,
                    "Fin": end
                })
            
            # Crear y mostrar dataframe interactivo
            df_inter = pd.DataFrame(intervalos)
            st.dataframe(
                df_inter[["Intervalo", "Punto de Prueba", "f''(x) Evaluada", "Estado Analítico"]], 
                use_container_width=True, 
                hide_index=True
            )
            
            st.divider()

            # --- 4. CONSTRUCCIÓN DEL GRÁFICO DINÁMICO (ColorFunction & Epilog / Prolog) ---
            st.header("4. Representación Gráfica Interactiva")
            fig = go.Figure()
            
            # Dibujar las curvas segmentadas tramo por tramo (ColorFunction)
            for inter in intervalos:
                x_vals = np.linspace(inter["Inicio"], inter["Fin"], 100)
                y_vals = []
                
                # Evaluación segura para evitar discontinuidades analíticas de división por cero
                for val in x_vals:
                    try:
                        res = float(func.subs(x, val).evalf())
                        y_vals.append(res)
                    except Exception:
                        y_vals.append(None)
                
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color=inter["Color"], width=4),
                    name=inter["Estado Analítico"].split()[-1],
                    hoverinfo='x+y'
                ))
            
            # Dibujar los puntos de inflexión con tooltips dinámicos (Prolog / Epilog)
            if puntos_inflexion_validos:
                puntos_inflexion_y = [float(func.subs(x, pt).evalf()) for pt in puntos_inflexion_validos]
                
                fig.add_trace(go.Scatter(
                    x=puntos_inflexion_validos, 
                    y=puntos_inflexion_y,
                    mode='markers',
                    marker=dict(
                        color='black', 
                        size=11, 
                        line=dict(color='white', width=2.5)
                    ),
                    name='Punto de Inflexión',
                    text=[f"Punto de Inflexión<br>x: {pt:.2f}<br>y: {func.subs(x, pt).evalf():.2f}" for pt in puntos_inflexion_validos],
                    hoverinfo='text'
                ))
            
            # Configuración estética del plano cartesiano
            fig.update_layout(
                xaxis_title=f"Eje {var_str}",
                yaxis_title="f(x)",
                showlegend=False,
                template="plotly_white",
                height=550,
                hovermode="closest"
            )
            
            # Forzar líneas de los ejes x=0 e y=0 tradicionales en laboratorios
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinewidth=1.5, zerolinecolor='Black')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=True, zerolinewidth=1.5, zerolinecolor='Black')
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error analítico: {e}. Asegúrate de ingresar una función válida (ejemplo: `x**3 - 3*x`).")
