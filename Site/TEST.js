// Variables globales
let criteriaMap = []; // sera rempli depuis criteria.txt
let database = [];    // sera rempli depuis database.txt

// Fonction pour charger les critères depuis un fichier txt
async function loadCriteria() {
    const response = await fetch("criteria.txt");
    const text = await response.text();
    const lines = text.split("\n").map(l => l.trim()).filter(l => l);

    const numCriteria = parseInt(lines[0]); // première ligne = nombre de critères
    criteriaMap = lines.slice(1, 1 + numCriteria);

    console.log("Critères chargés :", criteriaMap);
}

// Fonction pour charger la base de données
async function loadDatabase() {
    const response = await fetch("database.txt");
    const text = await response.text();
    const lines = text.split("\n").map(l => l.trim()).filter(l => l);

    database = lines.map(line => {
        const [url, name, score, boolStr] = line.split(";");
        return {
            url,
            name,
            score: parseInt(score),
            bools: boolStr.split(",").map(b => parseInt(b))
        };
    });

    console.log("Base chargée :", database);
}

// Fonction de recherche d'un site ou entreprise
async function searchSite() {
    const input = document.getElementById("searchInput").value.toLowerCase();
    const resultBox = document.getElementById("result");
    const siteName = document.getElementById("siteName");
    const scoreValue = document.getElementById("scoreValue");
    const scoreLabel = document.getElementById("scoreLabel");
    const criteriaList = document.getElementById("criteriaList");

    // Charger les fichiers si pas encore fait
    if (criteriaMap.length === 0) await loadCriteria();
    if (database.length === 0) await loadDatabase();

    let found = false;

    database.forEach(entry => {
        if (entry.name.toLowerCase().includes(input) || entry.url.toLowerCase().includes(input)) {
            found = true;

            // Affiche nom et lien
            siteName.innerHTML = `<a href="${entry.url}" target="_blank">${entry.name}</a>`;

            // Détermine label et couleur en fonction du score
            let label, colorClass;
            const score = entry.score;
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

            scoreValue.textContent = score;
            scoreValue.style.color = colorClass;
            scoreLabel.textContent = label;
            scoreLabel.style.color = colorClass;

            // Liste des critères pénalisants
            criteriaList.innerHTML = "";
            entry.bools.forEach((b, i) => {
                if (b === 1) {
                    const li = document.createElement("li");
                    li.textContent = criteriaMap[i];
                    criteriaList.appendChild(li);
                }
            });
        }
    });

    if (!found) {
        siteName.textContent = "Aucun résultat trouvé";
        scoreValue.textContent = "";
        scoreLabel.textContent = "";
        criteriaList.innerHTML = "";
    }

    resultBox.classList.remove("hidden");
}
