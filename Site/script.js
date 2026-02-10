let criteriaMap = []; // Liste des critères
let siteDatabase = {}; // Base des sites et critères

// --------------------------
// Charger les critères depuis criteria.txt
// --------------------------
async function loadCriteria() {
    try {
        const response = await fetch('criteria.txt');
        if (!response.ok) throw new Error("Impossible de charger le fichier criteria.txt");

        const text = await response.text();
        const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);
        const nbCriteria = parseInt(lines[0], 10);
        criteriaMap = lines.slice(1, nbCriteria + 1);
        fillCriteriaList(); // mise à jour page criteres.html
        await loadSites();   // charger la base des sites après critères
    } catch (err) {
        console.error("Erreur chargement critères :", err);
    }
}

// --------------------------
// Charger la base des sites depuis sites.txt
// Format : lien;nom;score;crit1,crit2,...
// --------------------------
async function loadSites() {
    try {
        const response = await fetch("sites.txt");
        if (!response.ok) throw new Error("Impossible de charger le fichier sites.txt");

        const text = await response.text();
        const lines = text.split("\n").map(l => l.trim()).filter(l => l.length > 0);

        lines.forEach(line => {
            const parts = line.split(";");
            const url = parts[0];
            const name = parts[1];
            const score = parseFloat(parts[2]); // score float
            const criteriaFlags = parts[3].split(",").map(s => s.trim() === "true" || s.trim() === "1");
            siteDatabase[url] = {name, criteriaFlags, score};
        });
    } catch (err) {
        console.error("Erreur chargement sites :", err);
    }
}

// --------------------------
// Obtenir label de score
// --------------------------
function getScoreLabel(score) {
    let label = "";
    let colorClass = "";

    if (score >= 8.0) {
        label = "🟢 Faible invasivité";
        colorClass = "green";
    } else if (score >= 5.0) {
        label = "🟠 Transparence moyenne";
        colorClass = "orange";
    } else {
        label = "🔴 Risque élevé";
        colorClass = "red";
    }

    return {label, colorClass};
}

// --------------------------
// Recherche d'un site
// --------------------------
function searchSite() {
    const input = document.getElementById("searchInput").value.trim().toLowerCase();
    if (!input) return alert("Veuillez entrer un nom de site ou URL");

    let siteData = null;
    let siteKey = null;
    for (const key in siteDatabase) {
        const data = siteDatabase[key];
        if (key.toLowerCase() === input || data.name.toLowerCase() === input) {
            siteData = data;
            siteKey = key;
            break;
        }
    }

    if (!siteData) return alert("Site non trouvé dans la base");

    const resultSection = document.getElementById("result");
    document.getElementById("siteName").textContent = siteData.name + " (" + siteKey + ")";
    document.getElementById("scoreValue").textContent = siteData.score.toFixed(1) + "/10"; // float avec 1 décimale

    const scoreInfo = getScoreLabel(siteData.score);
    const scoreLabel = document.getElementById("scoreLabel");
    scoreLabel.textContent = scoreInfo.label;
    scoreLabel.className = scoreInfo.colorClass;

    // Afficher les critères pénalisants
    const ul = document.getElementById("criteriaList");
    ul.innerHTML = "";
    siteData.criteriaFlags.forEach((c, i) => {
        if (c) {
            const li = document.createElement("li");
            li.textContent = criteriaMap[i];
            ul.appendChild(li);
        }
    });

    resultSection.classList.remove("hidden");
}

// --------------------------
// Remplir la page criteres.html
// --------------------------
function fillCriteriaList() {
    const ul = document.getElementById("criteriaFullList");
    if (!ul) return;
    ul.innerHTML = "";
    criteriaMap.forEach(crit => {
        const li = document.createElement("li");
        li.textContent = crit;
        ul.appendChild(li);
    });
}

// --------------------------
// Initialisation
// --------------------------
document.addEventListener("DOMContentLoaded", loadCriteria);
