// Three.js Background Animation
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('bg-canvas'), alpha: true });

renderer.setSize(window.innerWidth, window.innerHeight);
camera.position.z = 5;

// Particle System
const particlesGeometry = new THREE.BufferGeometry();
const particlesCount = 5000;
const posArray = new Float32Array(particlesCount * 3);

for (let i = 0; i < particlesCount * 3; i++) {
    posArray[i] = (Math.random() - 0.5) * 50;
}

particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMaterial = new THREE.PointsMaterial({
    size: 0.05,
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.8
});

const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
scene.add(particlesMesh);

// Floating Earth on Home Page
let earthMesh;
const earthGeometry = new THREE.SphereGeometry(2, 32, 32);
const earthMaterial = new THREE.MeshPhongMaterial({
    color: 0x00d4ff,
    wireframe: true,
    transparent: true,
    opacity: 0.6
});
earthMesh = new THREE.Mesh(earthGeometry, earthMaterial);

// Animated Satellite
let satelliteMesh;
const satelliteGroup = new THREE.Group();

// Satellite body
const satelliteBodyGeometry = new THREE.BoxGeometry(0.3, 0.2, 0.4);
const satelliteBodyMaterial = new THREE.MeshPhongMaterial({ color: 0xffffff });
const satelliteBody = new THREE.Mesh(satelliteBodyGeometry, satelliteBodyMaterial);

// Solar panels
const panelGeometry = new THREE.PlaneGeometry(0.8, 0.3);
const panelMaterial = new THREE.MeshPhongMaterial({ color: 0x1a1a2e, side: THREE.DoubleSide });
const leftPanel = new THREE.Mesh(panelGeometry, panelMaterial);
const rightPanel = new THREE.Mesh(panelGeometry, panelMaterial);

leftPanel.position.set(-0.6, 0, 0);
rightPanel.position.set(0.6, 0, 0);
leftPanel.rotation.y = Math.PI / 2;
rightPanel.rotation.y = Math.PI / 2;

satelliteGroup.add(satelliteBody);
satelliteGroup.add(leftPanel);
satelliteGroup.add(rightPanel);
satelliteMesh = satelliteGroup;

const light = new THREE.PointLight(0xffffff, 1, 100);
light.position.set(10, 10, 10);
scene.add(light);

// Globe for Explore Page
let globeMesh;
const globeGeometry = new THREE.SphereGeometry(3, 64, 64);
const globeMaterial = new THREE.MeshPhongMaterial({
    color: 0x7b2ff7,
    wireframe: false,
    transparent: true,
    opacity: 0.8
});
globeMesh = new THREE.Mesh(globeGeometry, globeMaterial);

function animate() {
    requestAnimationFrame(animate);
    
    particlesMesh.rotation.y += 0.001;
    particlesMesh.rotation.x += 0.0005;
    
    if (earthMesh.parent) {
        earthMesh.rotation.y += 0.005;
        earthMesh.rotation.x += 0.002;
        
        // Animate satellite orbiting Earth
        if (satelliteMesh.parent) {
            const time = Date.now() * 0.001;
            satelliteMesh.position.x = Math.cos(time * 0.5) * 4;
            satelliteMesh.position.z = Math.sin(time * 0.5) * 4;
            satelliteMesh.position.y = Math.sin(time * 0.3) * 1;
            satelliteMesh.lookAt(earthMesh.position);
            satelliteMesh.rotation.z += 0.01;
        }
    }
    
    if (globeMesh.parent) {
        globeMesh.rotation.y += 0.003;
    }
    
    renderer.render(scene, camera);
}

animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Navigation
function navigateTo(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(link => link.classList.remove('active'));
    
    document.getElementById(pageId).classList.add('active');
    document.querySelector(`[href="#${pageId}"]`).classList.add('active');
    
    // Handle 3D objects per page
    if (pageId === 'home') {
        scene.remove(globeMesh);
        scene.add(earthMesh);
        scene.add(satelliteMesh);
        earthMesh.position.set(0, 0, 0);
        satelliteMesh.position.set(4, 0, 0);
    } else if (pageId === 'explore') {
        scene.remove(earthMesh);
        scene.remove(satelliteMesh);
        scene.add(globeMesh);
        globeMesh.position.set(0, 0, 0);
    } else {
        scene.remove(earthMesh);
        scene.remove(satelliteMesh);
        scene.remove(globeMesh);
    }
}

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const pageId = link.getAttribute('href').substring(1);
        navigateTo(pageId);
    });
});

// Initialize with earth and satellite on home page
scene.add(earthMesh);
scene.add(satelliteMesh);
earthMesh.position.set(0, 0, 0);
satelliteMesh.position.set(4, 0, 0);

// Upload & Classification Logic
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImg = document.getElementById('preview-img');
const removeBtn = document.getElementById('remove-btn');
const imageName = document.getElementById('image-name');
const imageSize = document.getElementById('image-size');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
let currentFile = null;

uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = '#00d4ff';
    uploadZone.style.background = 'rgba(0, 212, 255, 0.1)';
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.style.borderColor = 'rgba(0, 212, 255, 0.5)';
    uploadZone.style.background = 'rgba(255, 255, 255, 0.05)';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'rgba(0, 212, 255, 0.5)';
    uploadZone.style.background = 'rgba(255, 255, 255, 0.05)';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImageUpload(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImageUpload(file);
    }
});

removeBtn.addEventListener('click', () => {
    currentFile = null;
    previewSection.style.display = 'none';
    resultsSection.style.display = 'none';
    fileInput.value = '';
});

function handleImageUpload(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imageName.textContent = `📁 ${file.name}`;
        imageSize.textContent = `📏 Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        previewSection.style.display = 'block';
        resultsSection.style.display = 'none';
        previewSection.scrollIntoView({ behavior: 'smooth' });
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    analyzeBtn.innerHTML = '🔄 Analyzing...';
    analyzeBtn.disabled = true;
    
    // Always use demo mode for Vercel deployment
    const classes = ['desert', 'green_area', 'water', 'cloudy'];
    const randomClass = classes[Math.floor(Math.random() * classes.length)];
    const confidence = (Math.random() * 0.3 + 0.7) * 100;
    
    // Simulate processing time
    setTimeout(() => {
        displayResults(randomClass, confidence);
        analyzeBtn.innerHTML = '🔍 Analyze Image';
        analyzeBtn.disabled = false;
    }, 2000);
});

function displayResults(className, confidence) {
    const resultLabel = document.getElementById('result-label');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceText = document.getElementById('confidence-text');
    
    const classNames = {
        'desert': '🏜️ Desert',
        'green_area': '🌲 Green Area',
        'water': '🌊 Water',
        'cloudy': '☁️ Cloudy'
    };
    
    resultLabel.textContent = classNames[className] || className;
    confidenceFill.style.width = confidence + '%';
    confidenceText.textContent = `Confidence: ${confidence.toFixed(2)}%`;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Class Card Interactions
document.querySelectorAll('.class-card').forEach(card => {
    card.addEventListener('click', () => {
        const className = card.getAttribute('data-class');
        card.style.transform = 'scale(1.1) rotateY(10deg)';
        setTimeout(() => {
            card.style.transform = '';
        }, 300);
    });
});

// Mouse parallax effect
document.addEventListener('mousemove', (e) => {
    const x = (e.clientX / window.innerWidth - 0.5) * 0.5;
    const y = (e.clientY / window.innerHeight - 0.5) * 0.5;
    
    if (earthMesh.parent) {
        earthMesh.rotation.y += x * 0.01;
        earthMesh.rotation.x += y * 0.01;
    }
});