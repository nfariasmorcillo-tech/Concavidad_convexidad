import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configuración de la página web
st.set_page_config(page_title="Laboratorio de Optimización y Concavidad", page_icon="🧮", layout="wide")

# =========================================================================
# BARRA LATERAL: SELECCIÓN DEL LABORATORIO PRINCIPAL
# =========================================================================
st.sidebar.title("🧮 Menú de Herramientas")
modulo_principal = st.sidebar.selectbox(
    "Selecciona el laboratorio:",
    ("Optimización Multi-Variable (f(x,y))", "Análisis de Concavidad e Intervalos (f(x))")
)

st.sidebar.divider()

# =========================================================================
# MÓDULO 1: OPTIMIZACIÓN MULTI-VARIABLE (CÓDIGO ANTERIOR CONSOLIDADO)
# =========================================================================
if modulo_principal == "Optimización Multi-Variable (f(x,y))":
    st.title("🧮 Laboratorio de Optimización Matemática")
    st.markdown("Herramienta pedagógica para el análisis de óptimos locales con y sin restricciones.")

    st.sidebar.header("📥 Configuración del Ejercicio")
    tipo_opt = st.sidebar.radio(
        "Selecciona el tipo de optimización:",
        ("Optimización Libre (Sin restricciones)", "Optimización Condicionada (Con restricciones)")
)

    funcion_str = st.sidebar.text_input("Función Objetivo f(x,y):", value="x**2 + y**2 - 4*x - 6*y")
    variables_str = st.sidebar.text_input("Variables (separadas por coma):", value="x, y")

    restriccion_str = ""
    if tipo_opt == "Optimización Condicionada (Con restricciones)":
        restriccion_str = st.sidebar.text_input("Restricción g(x,y) = 0:", value="x + y - 4")

    st.sidebar.info("**Nota de sintaxis:**\n* Usar `*` para multiplicar.\n* Usar `**` para potencias.")
    calcular = st.sidebar.button("Calcular Óptimos", type="primary")

    if calcular:
        try:
            vars_list = [sp.Symbol(v.strip()) for v in variables_str.split(',') if v.strip()]
            if len(vars_list) != 2:
                st.error("Por favor, ingresa exactamente dos variables libres (ej: x, y).")
            else:
                x, y = vars_list[0], vars_list[1]
                f = sp.sympify(funcion_str)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.latex(rf"\text{{Función Objetivo: }} f({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(f)}")
                with col2:
                    if tipo_opt == "Optimización Condicionada (Con restricciones)":
                        g = sp.sympify(restriccion_str)
                        st.latex(rf"\text{{Restricción: }} g({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(g)} = 0")

                st.divider()

                if tipo_opt == "Optimización Libre (Sin restricciones)":
                    st.header("1. Primeras Derivadas y Sistema de Ecuaciones")
                    fx = sp.diff(f, x)
                    fy = sp.diff(f, y)
                    
                    c1, c2 = st.columns(2)
                    c1.latex(rf"f_x = \frac{{\partial f}}{{\partial x}} = {sp.latex(fx)}")
                    c2.latex(rf"f_y = \frac{{\partial f}}{{\partial y}} = {sp.latex(fy)}")
                    
                    st.subheader("Sistema a resolver para puntos estacionarios:")
                    sistema_libre = r"\begin{cases} " + sp.latex(fx) + r" = 0 \\ " + sp.latex(fy) + r" = 0 \end{cases}"
                    st.latex(sistema_libre)
                    
                    puntos = sp.solve([fx, fy], (x, y), dict=True)
                    
                    with st.expander("🔍 Ver desarrollo algebraico paso a paso"):
                        st.markdown("### **Paso 1: Despejar de la primera ecuación**")
                        st.markdown(f"Tomamos $f_x = 0$ y aislamos la variable ${sp.latex(x)}$:")
                        st.latex(rf"{sp.latex(fx)} = 0")
                        despeje_x = sp.solve(fx, x)
                        if despeje_x:
                            st.markdown(f"Al resolver para ${sp.latex(x)}$, encontramos la relación:")
                            st.latex(rf"{sp.latex(x)} = {sp.latex(despeje_x[0])}")
                            st.markdown("### **Paso 2: Reemplazo en la segunda ecuación**")
                            fy_sustituida = fy.subs(x, despeje_x[0])
                            st.latex(rf"{sp.latex(fy)} = 0 \implies {sp.latex(fy_sustituida)} = 0")
                            sol_y = sp.solve(fy_sustituida, y)
                            st.latex(rf"{sp.latex(y)} = {sp.latex(sol_y)}")
                        
                        st.markdown("### **Paso 3: Coordenadas de los puntos críticos hallados**")
                        for idx, p in enumerate(puntos):
                            st.latex(rf"\text{{Punto }}{idx+1}: \quad \left( {sp.latex(p[x])}, \, {sp.latex(p[y])} \right)")

                    st.metric(label="Cantidad de puntos críticos encontrados", value=len(puntos))
                    
                    st.header("2. Segundas Derivadas y Matriz Hessiana")
                    fxx = sp.diff(fx, x)
                    fyy = sp.diff(fy, y)
                    fxy = sp.diff(fx, y)
                    H_gen = sp.Matrix([[fxx, fxy], [fxy, fyy]])
                    st.latex(rf"\text{{Matriz Hessiana (H): }} {sp.latex(H_gen)}")
                    
                    if len(puntos) > 0:
                        st.header("3. Tabla de Clasificación de Óptimos")
                        datos_tabla = []
                        for i, p in enumerate(puntos):
                            det_p = H_gen.det().subs(p)
                            fxx_p = fxx.subs(p)
                            val_f = f.subs(p)
                            tipo = "🟢 Mínimo Local" if det_p > 0 and fxx_p > 0 else "🔵 Máximo Local" if det_p > 0 else "🔴 Punto Silla" if det_p < 0 else "⚪ No decide"
                            datos_tabla.append({
                                "ID": f"P{i+1}", "Punto (x, y) Analítico": f"({sp.latex(p[x])}, {sp.latex(p[y])})",
                                "Punto (x, y) Decimal": f"({float(p[x].evalf()):.2f}, {float(p[y].evalf()):.2f})",
                                "f_xx (H1)": str(fxx_p), "|H| (H2)": f"{float(det_p.evalf()):.2f}",
                                "Tipo de Óptimo": tipo, "Valor f(x,y)": f"{float(val_f.evalf()):.2f}"
                            })
                        st.dataframe(pd.DataFrame(datos_tabla), use_container_width=True, hide_index=True)

                # Optimización condicionada (Lagrange con Factorización)
                else:
                    lam = sp.Symbol('λ')
                    L = f - lam * g
                    st.header("1. Función del Lagrangiano")
                    st.latex(rf"\mathcal{{L}}({sp.latex(x)}, {sp.latex(y)}, \lambda) = {sp.latex(L)}")
                    
                    gx = sp.diff(g, x)
                    gy = sp.diff(g, y)
                    Lx = sp.diff(L, x)
                    Ly = sp.diff(L, y)
                    Llam = sp.diff(L, lam)
                    
                    puntos = sp.solve([Lx, Ly, Llam], (x, y, lam), dict=True)
                    
                    with st.expander("🔍 Ver desarrollo algebraico paso a paso del Lagrangiano"):
                        st.markdown("### **Paso 1: Expresar u obtener el valor de $\lambda$**")
                        sol_lam_x = sp.solve(Lx, lam)
                        sol_lam_y = sp.solve(Ly, lam)
                        if sol_lam_x and sol_lam_y:
                            st.latex(rf"\text{{De }}\mathcal{{L}}_x=0: \lambda = {sp.latex(sol_lam_x[0])} \quad | \quad \text{{De }}\mathcal{{L}}_y=0: \lambda = {sp.latex(sol_lam_y[0])}")
                            st.markdown("### **Paso 2: Igualación de expresiones y Factorización**")
                            num_x, den_x = sp.fraction(sol_lam_x[0])
                            num_y, den_y = sp.fraction(sol_lam_y[0])
                            ecuacion_cruzada = num_x * den_y - num_y * den_x
                            ecuacion_factorizada = sp.factor(ecuacion_cruzada)
                            st.latex(rf"{sp.latex(ecuacion_cruzada)} = 0 \implies {sp.latex(ecuacion_factorizada)} = 0")
                            
                            despejes_y = sp.solve(ecuacion_factorizada, y)
                            for d_y in despejes_y:
                                st.latex(rf"y = {sp.latex(d_y)}")

                    st.header("2. Resultados del Criterio del Hessiano Orlado")
                    HOR = sp.Matrix([[0, gx, gy], [gx, sp.diff(Lx, x), sp.diff(Lx, y)], [gy, sp.diff(Lx, y), sp.diff(Ly, y)]])
                    
                    datos_tabla = []
                    for i, p in enumerate(puntos):
                        det_p = HOR.det().subs(p)
                        val_f = f.subs({x: p[x], y: p[y]})
                        tipo = "🔵 Máximo condicionado" if det_p > 0 else "🟢 Mínimo condicionado"
                        datos_tabla.append({
                            "ID": f"P{i+1}", "Punto (x, y)": f"({float(p[x].evalf()):.2f}, {float(p[y].evalf()):.2f})",
                            "Valor λ": f"{float(p[lam].evalf()):.2f}", "|HOR| (Det)": f"{float(det_p.evalf()):.2f}",
                            "Tipo de Óptimo": tipo, "Valor f(x,y)": f"{float(val_f.evalf()):.2f}"
                        })
                    st.dataframe(pd.DataFrame(datos_tabla), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error matemático: {e}")

# =========================================================================
# MÓDULO 2: ANÁLISIS DE CONCAVIDAD E INTERVALOS (ConcavityIntervalsPlot)
# =========================================================================
else:
    st.title("📉 Análisis de Concavidad e Intervalos Gráficos ($f(x)$)")
    st.markdown("Determina analíticamente la concavidad (hacia arriba o hacia abajo) identificando cambios de signo mediante la segunda derivada.")

    st.sidebar.header("📥 Configuración de la Función")
    func_str = st.sidebar.text_input("Función f(x):", value="x**4 - 6*x**2 + 5")
    var_str = st.sidebar.text_input("Variable autónoma:", value="x")
    
    col_range1, col_range2 = st.sidebar.columns(2)
    xmin = col_range1.number_input("x mín:", value=-3.0)
    xmax = col_range2.number_input("x máx:", value=3.0)

    st.sidebar.info("**Código de color matemático utilizado:**\n* 🟢 Verde: Concavidad hacia arriba ($f''(x) > 0$).\n* 🟠 Naranja: Concavidad hacia abajo ($f''(x) < 0$).")
    analizar = st.sidebar.button("Analizar Concavidad", type="primary")

    if analizar:
        try:
            x = sp.Symbol(var_str.strip())
            func = sp.sympify(func_str)
            
            # Restricción explícita de trigonométricas pedida en Mathematica
            if any(func.has(trig) for trig in [sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc]):
                st.error("Error: La función contiene términos trigonométricos no lineales restringidos para este módulo analítico.")
            else:
                # 1. Calcular derivadas analíticas
                fp1 = sp.diff(func, x)
                fp2 = sp.diff(fp1, x)   # Segunda derivada analítica f''(x)
                
                st.header("1. Estudio Analítico de Derivadas")
                c_der1, c_der2 = st.columns(2)
                c_der1.latex(rf"f'(x) = \frac{{df}}{{dx}} = {sp.latex(fp1)}")
                c_der2.latex(rf"f''(x) = \frac{{d^2f}}{{dx^2}} = {sp.latex(fp2)}")
                
                # 2. Puntos críticos de concavidad (num y den para evitar pérdidas)
                tofp = sp.together(fp2)
                num, den = sp.fraction(tofp)
                
                # Encontrar ceros del numerador y denominador reales
                raices_num = sp.solve(sp.Eq(num, 0), x)
                raices_den = sp.solve(sp.Eq(den, 0), x)
                
                # Filtrar solo valores reales numéricos
                cpn = [float(r.evalf()) for r in raices_num if r.is_real]
                cpd = [float(r.evalf()) for r in raices_den if r.is_real]
                
                # Puntos críticos totales ordenados dentro del rango visible
                puntos_criticos = sorted(list(set(cpn + cpd)))
                puntos_en_rango = [p for p in puntos_criticos if xmin <= p <= xmax]
                
                st.subheader("Puntos de inflexión candidatos / Discontinuidades:")
                if puntos_criticos:
                    st.latex(rf"\text{{Puntos críticos encontrados para }} f''(x): x \in {sp.latex(puntos_criticos)}")
                else:
                    st.markdown("No se hallaron cambios algebraicos explícitos de concavidad en el plano real.")

                # 3. Construcción de intervalos de evaluación
                puntos_frontera = [xmin] + puntos_en_rango + [xmax]
                intervalos = []
                
                st.header("2. Análisis de Signo por Intervalos")
                
                for i in range(len(puntos_frontera) - 1):
                    start, end = puntos_frontera[i], puntos_frontera[i+1]
                    pmedio = (start + end) / 2.0
                    
                    # Evaluar signo en el punto medio del intervalo
                    valor_f2 = float(fp2.subs(x, pmedio).evalf())
                    
                    if valor_f2 > 0:
                        estado = "🟢 Concavidad Hacia Arriba (Cóncava)"
                        color_trama = "green"
                    elif valor_f2 < 0:
                        estado = "text-decoration: none; 🟠 Concavidad Hacia Abajo (Convexa)"
                        color_trama = "orange"
                    else:
                        estado = "⚪ Lineal / Inflexión constante"
                        color_trama = "gray"
                        
                    intervalos.append({"Inicio": start, "Fin": end, "Estado": estado, "Color": color_trama})
                
                # Mostrar al alumno la tabla analítica resumida
                df_inter = pd.DataFrame(intervalos)
                st.dataframe(df_inter[["Inicio", "Fin", "Estado"]], use_container_width=True, hide_index=True)
                
                # 4. Construcción del Gráfico Segmentado (Simulando Plot con ColorFunction)
                st.header("3. Gráfico Interactivo de Concavidad Segmentada")
                fig = go.Figure()
                
                # Dibujar las curvas tramo por tramo para cambiar el color según la concavidad
                for inter in intervalos:
                    x_vals = np.linspace(inter["Inicio"], inter["Fin"], 100)
                    y_vals = [float(func.subs(x, val).evalf()) for val in x_vals]
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='lines',
                        line=dict(color=inter["Color"], width=4),
                        name=inter["Estado"].split()[-1]
                    ))
                
                # Agregar marcadores para los puntos de inflexión (Prolog/Epilog en Mathematica)
                puntos_inflexion_y = [float(func.subs(x, pt).evalf()) for pt in puntos_en_rango]
                
                if puntos_en_rango:
                    fig.add_trace(go.Scatter(
                        x=puntos_en_rango, y=puntos_inflexion_y,
                        mode='markers',
                        marker=dict(color='black', size=10, line=dict(color='white', width=2)),
                        name='Puntos de Inflexión',
                        text=[f"Inflexión x={pt:.2f}" for pt in puntos_en_rango],
                        hoverinfo='text+x+y'
                    ))
                
                fig.update_layout(
                    xaxis_title=f"Variable {var_str}",
                    yaxis_title="f(x)",
                    showlegend=False,
                    template="plotly_white",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Verifica la sintaxis matemática de la función en una variable: {e}")