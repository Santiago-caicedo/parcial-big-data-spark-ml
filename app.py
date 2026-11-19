"""
App de Streamlit: predicción de deserción y éxito académico con Spark ML.
Examen de Big Data – Spark ML · Autor: Santiago Caicedo Ortiz (U00165112)

Flujo:
  1. Carga Spark y el PipelineModel guardado en ./modelo_final (en memoria con @st.cache_resource).
  2. Formulario (st.form) con las variables que usa el modelo y el botón "Predecir".
  3. Calcula las variables derivadas (mismas fórmulas del notebook).
  4. Convierte la fila en un DataFrame de Spark y aplica el modelo.
  5. Muestra la clase predicha (con el orden de labels del StringIndexer) y la probabilidad de cada clase.
"""
import json
import pathlib
import zipfile

import altair as alt
import pandas as pd
import streamlit as st

BASE = pathlib.Path(__file__).parent
RUTA_MODELO = BASE / "modelo_final"
RUTA_ZIP = BASE / "modelo_final.zip"
RUTA_BANNER = BASE / "assets" / "banner.png"

AUTOR = "Santiago Caicedo Ortiz"
AUTOR_ID = "U00165112"

# Orden por defecto de StringIndexer(stringOrderType="frequencyDesc"); se sobrescribe con metadata.json
LABELS_POR_DEFECTO = ["Graduate", "Dropout", "Enrolled"]

INFO_CLASES = {
    "Dropout": {"es": "Deserción", "color": "#DC2626", "icono": "⚠️",
                "texto": "El estudiante presenta un perfil con **alto riesgo de abandonar** sus estudios. "
                         "Se recomienda activar acompañamiento académico y financiero cuanto antes."},
    "Enrolled": {"es": "Sigue matriculado", "color": "#D97706", "icono": "⏳",
                 "texto": "El estudiante probablemente **siga matriculado sin graduarse aún** "
                          "(avance lento o rezago). Conviene hacer seguimiento a su progreso."},
    "Graduate": {"es": "Graduación", "color": "#16A34A", "icono": "🎓",
                 "texto": "El estudiante tiene un perfil **en camino a graduarse**. "
                          "Mantener el seguimiento habitual."},
}

# ---------------------------------------------------------------------------------------------
# Diccionarios de códigos (codificación del dataset original, Realinho et al. 2021)
# ---------------------------------------------------------------------------------------------
ESTADO_CIVIL = {1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a", 4: "Divorciado/a", 5: "Unión de hecho",
                6: "Separado/a legalmente"}
CURSOS = {1: "Tecnologías de producción de biocombustibles", 2: "Diseño de animación y multimedia",
          3: "Servicio social (nocturno)", 4: "Agronomía", 5: "Diseño de comunicación", 6: "Enfermería veterinaria",
          7: "Ingeniería informática", 8: "Equinocultura", 9: "Administración", 10: "Servicio social",
          11: "Turismo", 12: "Enfermería", 13: "Higiene oral", 14: "Gestión de publicidad y marketing",
          15: "Periodismo y comunicación", 16: "Educación básica", 17: "Administración (nocturno)"}
MODO_INGRESO = {1: "1.ª fase – contingente general", 2: "Ordenanza 612/93",
                3: "1.ª fase – contingente especial (Azores)", 4: "Titulares de otros cursos superiores",
                5: "Ordenanza 854-B/99", 6: "Estudiante internacional (pregrado)",
                7: "1.ª fase – contingente especial (Madeira)", 8: "2.ª fase – contingente general",
                9: "3.ª fase – contingente general", 10: "Ordenanza 533-A/99 b2 (plan diferente)",
                11: "Ordenanza 533-A/99 b3 (otra institución)", 12: "Mayores de 23 años", 13: "Transferencia",
                14: "Cambio de carrera", 15: "Diploma de especialización tecnológica",
                16: "Cambio de institución/carrera", 17: "Diploma de ciclo corto",
                18: "Cambio de institución/carrera (internacional)"}
FORMACION_PREVIA = {1: "Educación secundaria", 2: "Licenciatura (bachelor)", 3: "Grado universitario",
                    4: "Maestría", 5: "Doctorado", 6: "Educación superior (sin terminar)",
                    7: "12.º año no completado", 8: "11.º año no completado", 9: "Otro – 11.º año",
                    10: "10.º año", 11: "10.º año no completado", 12: "Educación básica 3.er ciclo",
                    13: "Educación básica 2.º ciclo", 14: "Curso de especialización tecnológica",
                    15: "Educación superior – grado (1.er ciclo)", 16: "Curso técnico superior profesional",
                    17: "Educación superior – maestría (2.º ciclo)"}
NACIONALIDAD = {1: "Portuguesa", 2: "Alemana", 3: "Española", 4: "Italiana", 5: "Neerlandesa", 6: "Inglesa",
                7: "Lituana", 8: "Angoleña", 9: "Caboverdiana", 10: "Guineana", 11: "Mozambiqueña",
                12: "Santotomense", 13: "Turca", 14: "Brasileña", 15: "Rumana", 16: "Moldava", 17: "Mexicana",
                18: "Ucraniana", 19: "Rusa", 20: "Cubana", 21: "Colombiana"}


def codigos(n):
    return {i: f"Código {i}" for i in range(1, n + 1)}


SI_NO = {1: "Sí", 0: "No"}

# Definición de cada campo del formulario: (grupo, etiqueta, tipo, parámetros)
#   tipo "cat": selectbox de códigos · "bin": radio 0/1 · "int"/"float": number_input
CAMPOS = {
    # Datos personales
    "gender": ("personal", "Género", "bin", {"opciones": {1: "Masculino", 0: "Femenino"}, "defecto": 0}),
    "age_at_enrollment": ("personal", "Edad al matricularse", "int", {"min": 17, "max": 70, "defecto": 20}),
    "marital_status": ("personal", "Estado civil", "cat", {"opciones": ESTADO_CIVIL, "defecto": 1}),
    "nacionality": ("personal", "Nacionalidad", "cat", {"opciones": NACIONALIDAD, "defecto": 1}),
    "international": ("personal", "¿Estudiante internacional?", "bin", {"opciones": SI_NO, "defecto": 0}),
    "displaced": ("personal", "¿Desplazado de su ciudad de origen?", "bin", {"opciones": SI_NO, "defecto": 1}),
    "educational_special_needs": ("personal", "¿Necesidades educativas especiales?", "bin",
                                  {"opciones": SI_NO, "defecto": 0}),
    # Admisión y programa
    "course": ("admision", "Programa (carrera)", "cat", {"opciones": CURSOS, "defecto": 12}),
    "application_mode": ("admision", "Modalidad de ingreso", "cat", {"opciones": MODO_INGRESO, "defecto": 1}),
    "application_order": ("admision", "Orden de preferencia de la solicitud (0 = primera opción)", "int",
                          {"min": 0, "max": 9, "defecto": 1}),
    "daytime_evening_attendance": ("admision", "Jornada", "bin",
                                   {"opciones": {1: "Diurna", 0: "Nocturna"}, "defecto": 1}),
    "previous_qualification": ("admision", "Formación previa", "cat", {"opciones": FORMACION_PREVIA, "defecto": 1}),
    # Familia
    "mother_s_qualification": ("familia", "Nivel educativo de la madre", "cat", {"opciones": codigos(29), "defecto": 1}),
    "father_s_qualification": ("familia", "Nivel educativo del padre", "cat", {"opciones": codigos(34), "defecto": 27}),
    "mother_s_occupation": ("familia", "Ocupación de la madre", "cat", {"opciones": codigos(32), "defecto": 10}),
    "father_s_occupation": ("familia", "Ocupación del padre", "cat", {"opciones": codigos(46), "defecto": 10}),
    # Situación económica
    "tuition_fees_up_to_date": ("economia", "¿Matrícula al día?", "bin", {"opciones": SI_NO, "defecto": 1}),
    "debtor": ("economia", "¿Tiene deudas con la universidad?", "bin", {"opciones": SI_NO, "defecto": 0}),
    "scholarship_holder": ("economia", "¿Tiene beca?", "bin", {"opciones": SI_NO, "defecto": 0}),
    # Contexto macroeconómico (año de ingreso)
    "unemployment_rate": ("macro", "Tasa de desempleo (%)", "float", {"min": 0.0, "max": 30.0, "defecto": 11.1}),
    "inflation_rate": ("macro", "Tasa de inflación (%)", "float", {"min": -5.0, "max": 10.0, "defecto": 1.4}),
    "gdp": ("macro", "Crecimiento del PIB (%)", "float", {"min": -10.0, "max": 10.0, "defecto": 0.32}),
}
for s, nombre in (("1st", "1s"), ("2nd", "2s")):
    CAMPOS.update({
        f"curricular_units_{s}_sem_enrolled": (nombre, "Materias inscritas", "int", {"min": 0, "max": 30, "defecto": 6}),
        f"curricular_units_{s}_sem_evaluations": (nombre, "Materias presentadas a evaluación", "int",
                                                   {"min": 0, "max": 50, "defecto": 8}),
        f"curricular_units_{s}_sem_approved": (nombre, "Materias aprobadas", "int", {"min": 0, "max": 30, "defecto": 5}),
        f"curricular_units_{s}_sem_grade": (nombre, "Nota promedio (0 a 20)", "float", {"min": 0.0, "max": 20.0, "defecto": 12.0}),
        f"curricular_units_{s}_sem_credited": (nombre, "Materias homologadas", "int", {"min": 0, "max": 25, "defecto": 0}),
        f"curricular_units_{s}_sem_without_evaluations": (nombre, "Materias sin evaluación", "int",
                                                          {"min": 0, "max": 15, "defecto": 0}),
    })

GRUPOS = {
    "personal": "👤 Datos personales",
    "admision": "🏫 Admisión y programa",
    "familia": "👪 Familia",
    "economia": "💰 Situación económica",
    "1s": "📘 1.er semestre",
    "2s": "📗 2.º semestre",
    "macro": "🌍 Contexto económico",
}

# Columnas originales que se necesitan para calcular las variables derivadas
BASE_DERIVADAS = [f"curricular_units_{s}_sem_{v}" for s in ("1st", "2nd")
                  for v in ("enrolled", "approved", "grade", "evaluations")]


def calcular_variables_derivadas(v):
    """Mismas fórmulas que agregar_variables_derivadas() del notebook (división por cero -> 0)."""
    def tasa(num, den):
        return v[num] / v[den] if v[den] > 0 else 0.0
    return {
        "tasa_aprobacion_1s": tasa("curricular_units_1st_sem_approved", "curricular_units_1st_sem_enrolled"),
        "tasa_aprobacion_2s": tasa("curricular_units_2nd_sem_approved", "curricular_units_2nd_sem_enrolled"),
        "variacion_nota": float(v["curricular_units_2nd_sem_grade"] - v["curricular_units_1st_sem_grade"]),
        "eficiencia_evaluacion_2s": tasa("curricular_units_2nd_sem_approved", "curricular_units_2nd_sem_evaluations"),
    }


# ---------------------------------------------------------------------------------------------
# Spark y modelo (se cargan una sola vez y quedan en memoria)
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Iniciando Spark y cargando el modelo… (la primera vez tarda unos segundos)")
def cargar_spark_y_modelo():
    from pyspark.ml import PipelineModel
    from pyspark.sql import SparkSession

    if not RUTA_MODELO.exists() and RUTA_ZIP.exists():
        with zipfile.ZipFile(RUTA_ZIP) as z:
            z.extractall(BASE)

    spark = (SparkSession.builder
             .master("local[1]")
             .appName("app_desercion_sparkml")
             .config("spark.driver.memory", "1g")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "1")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    modelo = PipelineModel.load(str(RUTA_MODELO))

    meta = {}
    ruta_meta = RUTA_MODELO / "metadata.json"
    if ruta_meta.exists():
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    return spark, modelo, meta


def columnas_del_modelo(meta):
    """Columnas que el modelo espera recibir (sin las derivadas, que se calculan)."""
    derivadas = set(meta.get("derivadas", []))
    usadas = [c for c in meta.get("categoricas", []) + meta.get("numericas", []) if c not in derivadas]
    if not usadas:  # sin metadata: usar todas las variables del dataset
        usadas = list(CAMPOS)
    return usadas


def predecir(spark, modelo, fila):
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

    esquema = StructType([
        StructField(c, IntegerType() if CAMPOS.get(c, (None, None, "float"))[2] in ("cat", "bin", "int")
                    else DoubleType(), False)
        for c in fila
    ])
    valores = tuple(int(v) if isinstance(f.dataType, IntegerType) else float(v)
                    for v, f in zip(fila.values(), esquema.fields))
    sdf = spark.createDataFrame([valores], esquema)
    r = modelo.transform(sdf).select("prediction", "probability").first()
    return int(r["prediction"]), r["probability"].toArray().tolist()


# ---------------------------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------------------------
st.set_page_config(page_title="Predicción de Deserción Estudiantil", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label { font-family: 'Poppins', sans-serif; }
h1, h2, h3 { color: #0B1F3A; font-weight: 700; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
div[data-testid="stForm"] { background: #FFFFFF; border: 1px solid #D5E7E4; border-radius: 16px;
                            padding: 1.2rem 1.4rem; box-shadow: 0 4px 14px rgba(11,31,58,.06); }
div[data-testid="stFormSubmitButton"] button { width: 100%; font-size: 1.1rem; font-weight: 600;
                                               padding: .7rem; border-radius: 12px; }
.instrucciones { background: #E6F4F1; border-left: 6px solid #0F766E; border-radius: 12px;
                 padding: 1rem 1.3rem; margin: .8rem 0 1.2rem 0; color: #0B1F3A; }
.instrucciones ol { margin: .3rem 0 0 1.1rem; padding: 0; }
.resultado { border-radius: 18px; padding: 1.4rem 1.6rem; color: white; margin-bottom: 1rem;
             box-shadow: 0 6px 20px rgba(0,0,0,.12); }
.resultado .etq { font-size: .95rem; opacity: .9; text-transform: uppercase; letter-spacing: .08em; }
.resultado .clase { font-size: 2.3rem; font-weight: 700; line-height: 1.2; }
.resultado .prob { font-size: 1.05rem; opacity: .95; }
.pie { text-align: center; color: #475569; font-size: .92rem; margin-top: 3rem; padding: 1.2rem;
       border-top: 1px solid #CBD5E1; }
.pie b { color: #0F766E; }
</style>
""", unsafe_allow_html=True)

if RUTA_BANNER.exists():
    st.image(str(RUTA_BANNER))
st.title("🎓 Predicción de Deserción y Éxito Académico")
st.caption("Modelo de clasificación multiclase entrenado con **Spark ML** sobre el dataset "
           "*Predict Students' Dropout and Academic Success* (≈ 4.400 estudiantes).")

st.markdown("""
<div class="instrucciones">
<b>📋 Instrucciones de uso</b>
<ol>
  <li>Completa los datos del estudiante en cada pestaña del formulario (personales, admisión, familia,
      situación económica, rendimiento por semestre y contexto económico). Los valores iniciales son los más
      comunes en el dataset.</li>
  <li>En las pestañas de semestre, ingresa cuántas materias inscribió, presentó y aprobó, y su nota promedio
      (escala de 0 a 20). Las materias aprobadas no pueden superar a las inscritas.</li>
  <li>Pulsa <b>🔮 Predecir</b>. La app calcula las variables derivadas (tasas de aprobación y variación de la nota),
      aplica el modelo y muestra la <b>clase predicha</b> con la <b>probabilidad de cada clase</b>.</li>
</ol>
</div>
""", unsafe_allow_html=True)

try:
    spark, modelo, meta = cargar_spark_y_modelo()
except Exception as e:  # noqa: BLE001
    st.error("No se pudo cargar el modelo. Verifica que la carpeta `modelo_final` esté junto a `app.py` "
             f"y que Java esté instalado (`packages.txt` → `default-jre`).\n\nDetalle: {e}")
    st.stop()

LABELS = meta.get("labels", LABELS_POR_DEFECTO)
COLUMNAS = columnas_del_modelo(meta)
CAMPOS_FORM = [c for c in CAMPOS if c in COLUMNAS or c in BASE_DERIVADAS]

with st.sidebar:
    st.header("ℹ️ Sobre el modelo")
    st.markdown(f"**Algoritmo:** {meta.get('modelo', 'PipelineModel de Spark ML')}")
    if meta.get("hiperparametros"):
        st.markdown("**Mejores hiperparámetros:**")
        st.json(meta["hiperparametros"])
    if meta.get("metricas_test"):
        m = meta["metricas_test"]
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m.get('accuracy', 0):.1%}")
        c2.metric("F1 ponderado", f"{m.get('f1_ponderado', 0):.1%}")
    st.markdown("**Clases:**")
    for lbl in LABELS:
        info = INFO_CLASES.get(lbl, {})
        st.markdown(f"{info.get('icono', '•')} **{lbl}** – {info.get('es', '')}")
    st.caption("Las categorías siguen la codificación del dataset original (Realinho et al., 2021). "
               "El modelo usa Spark en modo local dentro de la app.")

# ---------------------------------------------------------------- formulario
valores = {}
with st.form("formulario_estudiante"):
    st.subheader("📝 Datos del estudiante")
    grupos_presentes = [g for g in GRUPOS if any(CAMPOS[c][0] == g for c in CAMPOS_FORM)]
    pestanas = st.tabs([GRUPOS[g] for g in grupos_presentes])
    for g, tab in zip(grupos_presentes, pestanas):
        with tab:
            campos_grupo = [c for c in CAMPOS_FORM if CAMPOS[c][0] == g]
            cols = st.columns(2)
            for i, c in enumerate(campos_grupo):
                _, etiqueta, tipo, p = CAMPOS[c]
                with cols[i % 2]:
                    if tipo == "cat":
                        opciones = list(p["opciones"])
                        valores[c] = st.selectbox(etiqueta, opciones, index=opciones.index(p["defecto"]),
                                                  format_func=lambda k, o=p["opciones"]: f"{k} · {o[k]}"
                                                  if not o[k].startswith("Código") else o[k], key=c)
                    elif tipo == "bin":
                        opciones = list(p["opciones"])
                        valores[c] = st.radio(etiqueta, opciones, index=opciones.index(p["defecto"]),
                                              format_func=lambda k, o=p["opciones"]: o[k], horizontal=True, key=c)
                    elif tipo == "int":
                        valores[c] = st.number_input(etiqueta, min_value=p["min"], max_value=p["max"],
                                                     value=p["defecto"], step=1, key=c)
                    else:
                        valores[c] = st.number_input(etiqueta, min_value=float(p["min"]), max_value=float(p["max"]),
                                                     value=float(p["defecto"]), step=0.1, format="%.2f", key=c)
    enviado = st.form_submit_button("🔮 Predecir", type="primary")

# ---------------------------------------------------------------- predicción
if enviado:
    errores = []
    for s, nombre in (("1st", "1.er"), ("2nd", "2.º")):
        if valores[f"curricular_units_{s}_sem_approved"] > valores[f"curricular_units_{s}_sem_enrolled"]:
            errores.append(f"{nombre} semestre: las materias aprobadas no pueden superar a las inscritas.")
    if errores:
        for e in errores:
            st.warning(e)
        st.stop()

    derivadas = calcular_variables_derivadas(valores)
    fila = {c: valores[c] for c in COLUMNAS if c in valores}
    fila.update({k: v for k, v in derivadas.items() if k in meta.get("numericas", derivadas)})

    with st.spinner("Aplicando el modelo de Spark…"):
        idx, probs = predecir(spark, modelo, fila)

    clase = LABELS[idx]
    info = INFO_CLASES.get(clase, {"es": clase, "color": "#0F766E", "icono": "•", "texto": ""})

    st.subheader("📊 Resultado de la predicción")
    col_izq, col_der = st.columns([1, 1.3], gap="large")
    with col_izq:
        st.markdown(f"""
        <div class="resultado" style="background: linear-gradient(135deg, {info['color']}, {info['color']}cc);">
          <div class="etq">Clase predicha</div>
          <div class="clase">{info['icono']} {clase}</div>
          <div class="prob">{info['es']} · probabilidad {probs[idx]:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(info["texto"])

        st.markdown("**Variables derivadas calculadas:**")
        d1, d2 = st.columns(2)
        d1.metric("Tasa aprobación 1.er sem", f"{derivadas['tasa_aprobacion_1s']:.0%}")
        d2.metric("Tasa aprobación 2.º sem", f"{derivadas['tasa_aprobacion_2s']:.0%}")
        d1.metric("Variación de la nota", f"{derivadas['variacion_nota']:+.2f}")
        d2.metric("Eficiencia evaluación 2.º sem", f"{derivadas['eficiencia_evaluacion_2s']:.0%}")

    with col_der:
        tabla = pd.DataFrame({
            "Clase": LABELS,
            "Significado": [INFO_CLASES.get(l, {}).get("es", "") for l in LABELS],
            "Probabilidad": probs,
        }).sort_values("Probabilidad", ascending=False)

        grafico = (alt.Chart(tabla)
                   .mark_bar(cornerRadiusEnd=6, height=34)
                   .encode(x=alt.X("Probabilidad:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                           y=alt.Y("Clase:N", sort="-x", title=None),
                           color=alt.Color("Clase:N", legend=None,
                                           scale=alt.Scale(domain=list(INFO_CLASES),
                                                           range=[v["color"] for v in INFO_CLASES.values()])),
                           tooltip=["Clase", "Significado", alt.Tooltip("Probabilidad:Q", format=".1%")]))
        texto = grafico.mark_text(align="left", dx=6, fontWeight="bold").encode(
            text=alt.Text("Probabilidad:Q", format=".1%"), color=alt.value("#0B1F3A"))
        st.altair_chart((grafico + texto).properties(height=200, title="Probabilidad por clase"),
                        use_container_width=True)

        tabla_pct = tabla.assign(Probabilidad=tabla["Probabilidad"] * 100)
        st.dataframe(tabla_pct, hide_index=True, use_container_width=True,
                     column_config={"Probabilidad": st.column_config.ProgressColumn(
                         "Probabilidad", format="%.1f %%", min_value=0.0, max_value=100.0)})

st.markdown(f"""
<div class="pie">
  Desarrollado por <b>{AUTOR}</b> · ID {AUTOR_ID}<br>
  Examen de Big Data – Spark ML · Clasificación multiclase con PySpark y Streamlit
</div>
""", unsafe_allow_html=True)
