# Rail Riot : modèles voxel PRIMAL

Modèles voxel générés par code (Python + numpy, Blender en ligne de commande), selon le guide
« Faire un modèle voxel pour Rail Riot, de A à Z ».

| Modèle | Référence | Taille (L × l × h, studs) | Triangles (contour compris) | Textures |
|---|---|---|---|---|
| `storm_serpent` | serpent d'orage bleu, wagon en bois | ≈ 44,5 × 12,8 × 29,4 | 24 656 | 3 × 1024² |
| `fire_dragon` | dragon de feu, wagon industriel | ≈ 45,0 × 9,9 × 22,1 | 28 598 | 3 × 1024² |

Version actuelle : **v3** (`output/<modèle>/v3/`). Les v1 et v2 sont gardées à côté pour comparer.

Petit cube 0,16 stud, gros cube 0,48 stud (classe « boss colossal »).

## Fichiers livrés (`output/<modèle>/v3/`)

- `<modèle>.glb` et `<modèle>.fbx` : un mesh par partie (et quelques morceaux quand une partie dépasse
  un atlas), pivot de chaque partie au centre de son articulation, 1 unité = 1 stud.
- `<modèle>_atlas1..3.png` : textures 1024 × 1024, chacune < 1 Mio (studs natifs, AO, arêtes claires).
- `<modèle>_manifest.json` : parties et parents, pivots, néons (position, taille, rotation, couleur),
  points d'attache des effets, PointLights conseillées, hauteur du sol, triangles.
- `renders/` : vues de face, 3/4, profil, arrière et gros plans de la tête. `<modèle>_planche.jpg` les regroupe.

## Import dans Roblox Studio

1. File > Import 3D > choisir le `.glb` (ou le `.fbx`).
2. Vérifier l'échelle : 1 unité du fichier = 1 stud (le modèle fait environ 44 studs de long).
3. Chaque mesh devient une MeshPart avec sa texture (atlas). Les pièces `Neon...`, `Eye...`, `Crystal...`,
   `Lantern...`, `Pipe...`, `Vent...`, etc. sont les néons : leur mettre Material = Neon
   (couleurs dans le manifeste), ou les recréer en Parts depuis le manifeste.
4. Le contour noir est une coque inversée dans chaque mesh : laisser **DoubleSided = false**.
5. `fire_dragon` : la partie `Lava` (la cuve) peut passer en Neon pour qu'elle brille.

Les coordonnées du manifeste sont en axes Roblox (Y en haut). L'avant du train est +X.

## Parties (animation)

- `storm_serpent` : Wagon > Body > (Tail1 > Tail2), (Neck1 > Neck2 > Head > Jaw).
- `fire_dragon` : Wagon > Body > (Lava), (Tail1 > Tail2 > Tail3), (Neck > Head > Jaw).

Les faces de jonction entre parties mobiles sont gardées des deux côtés et peintes en sombre.

## Choix techniques

- **Studs** : la tuile native (`assets/roblox_studs_tile.png`) est posée une fois par gros cube, seulement
  sur les carrés plats complets, jamais en dessous, ni sur les yeux, les dents, la bouche ou la lave.
- **Contour** : la partie dilatée d'un petit cube (somme de Minkowski avec un cube : faces poussées,
  coins poussés le long de chaque face) puis poussée jusqu'à 0,2 (0,16 pour le wagon), faces retournées.
  Il est retiré dans les orbites et devant les yeux (`NO_OUTLINE`), sinon il les cache en noir.
- **Atlas** : les faces du côté gauche réutilisent les texels de leur jumelle du côté droit (modèles
  symétriques), ce qui divise la surface de texture par deux. 37,5 px par stud.
- **Triangles** : faces du dessous du wagon supprimées (invisibles), cornes et défenses à section carrée
  le long de courbes planes (3 fois moins de triangles que des cornes rondes), masses en gros cubes.
- **Mâchoire ouverte du dragon** : profil en escalier aligné sur les axes (pas de rotation, les studs restent).

## Régénérer

```bash
cd railriot-models
python3 -m pipeline.build models.storm_serpent v2          # voxels, meshes, atlas, export, rendus
python3 -m pipeline.build models.fire_dragon v2 --quick     # rendus rapides
python3 -m pipeline.sheet output/fire_dragon/v2             # planche
```

Dépendances : Python 3 avec numpy, scipy et Pillow ; Blender 4 (avec numpy) ; pngquant.

## Pipeline (`pipeline/`)

- `vox.py` : formes (boîte, ellipsoïde, cylindre, RoundCone, Sweep, plaque extrudée, arc), échantillonnage sur
  les deux grilles, `add` / `cut` / `paint`, bitmaps, nettoyage des morceaux flottants, symétrie.
- `mesh.py` : faces exposées par groupe rigide, greedy meshing, miroir des quads, coque du contour.
- `texture.py` : AO par lancer de rayons dans la grille, studs, arêtes, packing de l'atlas.
- `fx.py` : peinture procédurale (planches, lave, rayures de danger, écailles, tôles).
- `build.py` : orchestration, manifeste. `blender_export.py` : glb/fbx et rendus Cycles.

## Historique

**v3** (wagons plus simples, animaux plus grands et plus détaillés)
- Dragon de feu : cuve plus longue à 5 côtes épaisses, 10 piques très hautes avec un cœur de flammes à part
  qui brille (partie `Flames`), tête plus haute sur un cou plus long, queue qui pend derrière le wagon,
  écailles peintes. Wagon : plus de caisses latérales ni de rebord, rambardes allégées, griffures peintes,
  défenses en gros cubes. Le wagon passe de 7 400 à 3 650 triangles.
- Serpent d'orage : corps plus massif (bosse, épaule), cou très épais à la base, volutes en relief
  (3 disques en escalier), cristaux hexagonaux pointus (partie `Crystals`, qui brille), crinière de nuages
  continue le long du cou, bande néon jusqu'à la gorge, écailles peintes. Wagon : défenses en gros cubes,
  lanternes sans montants, mousse peinte, os croisés retirés. Le wagon passe de 17 000 à 5 800 triangles.
- Pipeline : parties « à plat » (une pastille de couleur par matière et orientation, aucun texel : idéal
  pour ce qui passe en Neon), parties lumineuses dans les rendus, partage des texels avant/arrière
  tolérant (le wagon du serpent tient dans un seul mesh), décalage de groupes de formes, vue `ref`
  au même angle que les références.

**v2**
- Dragon de feu : tête refaite (museau plus court qui descend du front vers le nez, crâne plus bas, arcade
  en haut du visage, mâchoire inférieure plus profonde au dessous ivoire, dents triangulaires et crocs,
  couronne de cornes qui continue sur la nuque), gorge ivoire continue, tuyaux plus épais avec colliers,
  queue qui monte au-dessus des caisses puis redescend (7 piques), échelle, griffures creusées, 3 fenêtres
  par côté de cabine.
- Serpent d'orage : nuages regroupés en grosses masses à dessus plus clair (collier à la base du cou,
  milieu et haut du cou, grosse masse à l'arrière, crinière plus grande), cristaux en 2 pièces et éclats au
  pied du grand cristal, lanternes 1,4 fois plus grandes, touffes de mousse sur le bord, points d'attache
  `Lightning1..6` pour les éclairs.
- Pipeline : packing en « skyline » (atlas remplis à 93 %), nombre d'échantillons de rendu réglable.

**v1** : premiers modèles complets (pipeline, exports, rendus).
