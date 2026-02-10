let criteriaMap = []; // Liste des critères
let siteDatabase = {}; // Base des sites et critères

// --------------------------
// Charger les critères depuis criteres.txt
// --------------------------
async function loadCriteria() {
    try {
        const response = await fetch('criteria.txt');
        if (!response.ok) throw new Error("Impossible de charger le fichier criteres.txt");

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
            const criteriaFlags = parts.slice(2).map(s => s.trim() === "true");
            siteDatabase[url] = {name, criteriaFlags};
        });
    } catch (err) {
        console.error("Erreur chargement sites :", err);
    }
}

// --------------------------
// Calcul du score
// --------------------------
function calculateScore(selectedCriteria) {
    let score = 10 - selectedCriteria.filter(c => c).length;
    let label = "";
    let colorClass = "";

    if (score >= 8) {
        label = "🟢 Faible invasivité";
        colorClass = "green";
    } else if (score >= 5) {
        label = "🟠 Transparence moyenne";
        colorClass = "orange";
    } else {
        label = "🔴 Risque élevé";
        colorClass = "red";
    }

    return {score, label, colorClass};
}

// --------------------------
// Recherche d'un site
// --------------------------
function searchSite() {
    const input = document.getElementById("searchInput").value.trim();
    if (!input) return alert("Veuillez entrer un nom de site ou URL");

    let siteData = siteDatabase[input];
    if (!siteData) return alert("Site non trouvé dans la base");

    const selectedCriteria = siteData.criteriaFlags;
    const result = calculateScore(selectedCriteria);

    const resultSection = document.getElementById("result");
    document.getElementById("siteName").textContent = siteData.name + " (" + input + ")";
    document.getElementById("scoreValue").textContent = result.score + "/10";
    const scoreLabel = document.getElementById("scoreLabel");
    scoreLabel.textContent = result.label;
    scoreLabel.className = result.colorClass;

    // Afficher les critères pénalisants
    const ul = document.getElementById("criteriaList");
    ul.innerHTML = "";
    selectedCriteria.forEach((c, i) => {
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
