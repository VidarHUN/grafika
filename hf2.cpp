//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Váradi Richárd Tamás
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

//Material oosztály megvalósítva 8.3-s videó alapján
enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) { type = t; };
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}

};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}

};

//Forrás: raytrace.cpp (Tárgyhonlap)
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

//Forrás: raytrace.cpp (Tárgyhonlap)
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

//Forrás: raytrace.cpp (Tárgyhonlap)
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

//Forrás: bmemodel.pdf
struct Quadrics : Intersectable {
	mat4 Q;

	float f(vec4 r) { //r.w = 1
		return dot(r * Q, r);
	}

	vec3 gradf(vec4 r) { //r.w = 1
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}
};

//Forrás: http://reality.cs.ucl.ac.uk/projects/quadrics/pbg06.pdf
struct Paraboloid : public Quadrics {
	vec3 center;
	float radius;

	Paraboloid(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		Q = mat4(-30, 0, 0, 0,
				  0, 0, 0, -2,
				  0, 0, -30, 0,
				  0, -2, 0, 0.1);

		vec4 s(dist.x, dist.y, dist.z, 1);
		vec4 d(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float A = dot(d * Q, d);
		float B = dot(d * Q, s) + dot(s * Q, d);
		float C = dot(s * Q, s);
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	// t1 >= t2 for sure
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = dist + ray.dir * hit.t;

		if (hit.position.y < -0.5) {
			hit.t = t1;
			hit.position = dist + ray.dir * hit.t;
			if (hit.position.y < -0.5) {
				hit.t = -1;
				hit.position = dist + ray.dir * hit.t;
			}
		}

		vec3 u = gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1));
		hit.normal = u / length(u);
		hit.material = material;
		return hit;
	}
};

//Forrás: http://reality.cs.ucl.ac.uk/projects/quadrics/pbg06.pdf
struct Elipsoid : public Quadrics {
	vec3 center;
	float radius;
	bool cut = false;

	Elipsoid(const vec3& _center, float _radius, Material* _material, bool b) {
		center = _center;
		radius = _radius;
		material = _material;
		cut = b;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		if (cut) {
			Q = mat4(5, 0, 0, 0,
				0, 30, 0, 0,
				0, 0, 5, 0,
				0, 0, 0, -radius);
		}
		else {
			Q = mat4(10, 0, 0, 0,
				0, 4, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, -radius);
		}
		vec4 s(dist.x, dist.y, dist.z, 1);
		vec4 d(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float A = dot(d * Q, d);
		float B = dot(d * Q, s) + dot(s * Q, d);
		float C = dot(s * Q, s);
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	// t1 >= t2 for sure
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = dist + ray.dir * hit.t;

		if (cut) {
			if (hit.position.y > 0.85) {
				hit.t = t1;
				hit.position = dist + ray.dir * hit.t;
				if (hit.position.y > 0.85) {
					hit.t = -1;
					hit.position = dist + ray.dir * hit.t;
				}
			}
		}
		vec3 u = gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1));
		hit.normal = normalize(u);
		hit.material = material;
		if (cut) {
			if (dot(hit.normal, ray.dir) > 0)
				hit.normal = hit.normal * (-1);
		}
		return hit;
	}
};

//Forrás: http://reality.cs.ucl.ac.uk/projects/quadrics/pbg06.pdf
struct Hyperboloid : public Quadrics {
	vec3 center;
	float radius;
	bool up = false;

	Hyperboloid(const vec3& _center, float _radius, Material* _material, bool b) {
		center = _center;
		radius = _radius;
		material = _material;
		up = b;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float r2 = radius * radius;
		if (up) {
			Q = mat4(-1, 0, 0, 0,
				0, 0.4, 0, 0,
				0, 0, -1, 0,
				0, 0, 0, radius);
		}
		else {
			Q = mat4(-20, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, -20, 0,
					0, 0, 0, radius);
		}
		vec4 s(dist.x, dist.y, dist.z, 1);
		vec4 d(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float A = dot(d * Q, d);
		float B = dot(d * Q, s) + dot(s * Q, d);
		float C = dot(s * Q, s);
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	// t1 >= t2 for sure
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = dist + ray.dir * hit.t;

		if (up) {
			if (hit.position.y < 0 || hit.position.y >  2.5) {
				hit.t = t1;
				hit.position = dist + ray.dir * hit.t;
				if (hit.position.y < 0 || hit.position.y > 2.5) {
					hit.t = -1;
					hit.position = dist + ray.dir * hit.t;
				}
			}
		}
		else if (hit.position.y > 0.7 || hit.position.y < -0.3) {
			hit.t = t1;
			hit.position = dist + ray.dir * hit.t;
			if (hit.position.y > 0.7 || hit.position.y < -0.3) {
				hit.t = -1;
				hit.position = dist + ray.dir * hit.t;
			}
		}
		vec3 u = gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1));
		hit.normal = u / length(u);
		hit.material = material;
		return hit;
	}
};

//Forrás: raytrace.cpp (Tárgyhonlap)
class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		fov = _fov;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
};

//Forrás: raytrace.cpp (Tárgyhonlap)
struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() {
	float random = ((float)rand()) / (float)RAND_MAX;
	return 0.2 + random * 0.4;
}

const float epsilon = 0.0001f;

//Forrás: raytrace.cpp (Tárgyhonlap)
class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La, sky;
	std::vector<vec3> controlPoints;

public:
	void build() {
		vec3 eye = vec3(0, 0,2), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.3f, 0.2f, 0.1f);
		sky = vec3(0.529412f, 0.807843f, 0.921569f);

		vec3 lightDirection(1, 1, 1), Le(0.1, 0.1, 0.1);
		lights.push_back(new Light(lightDirection, Le));

		Material* roughMaterial = new RoughMaterial(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2), 50);
		Material* gold = new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		Material* silver = new ReflectiveMaterial(vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1));
		Material* green = new RoughMaterial(vec3(0.2f, 0.8f, 0.1f), vec3(2, 2, 2), 50);
		Material* blue = new RoughMaterial(vec3(0.2f, 0.2f, 0.8f), vec3(2, 2, 2), 50);

		objects.push_back(new Elipsoid(vec3(0.1, -0.5, 0), 0.5, gold, false));
		objects.push_back(new Elipsoid(vec3(0, 0, 0), 24, roughMaterial, true));
		objects.push_back(new Hyperboloid(vec3(0.6, -0.3, 1), 0.1, green, false));
		objects.push_back(new Paraboloid(vec3(-0.6, -0.1, 1), 0.1, blue));
		objects.push_back(new Hyperboloid(vec3(0, 0.85, 0),0.48, silver, true));

		for (int i = 0; i < 5; i++) {
			controlPoints.push_back(vec3(rnd(), 0.85, rnd()));
		}
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}


	//Forrás: Részben a 8.3 videó
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 3) return La;

		vec3 sun = lights[0]->Le;
		vec3 sunDir = lights[0]->direction;

		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
			return sky + sun * pow(dot(ray.dir, sunDir), 10);

		vec3 outRadiance(0, 0, 0);
		if (hit.material->type == ROUGH){
			outRadiance = hit.material->ka * La;
			for (int i = 0; i < controlPoints.size(); i++){
				float An = (0.4 * 0.4) * M_PI / controlPoints.size();
				float cosTheta = dot(vec3(0, -1, 0), normalize(hit.position - controlPoints[i]));
				float r2 = length(controlPoints[i] - hit.position) * length(controlPoints[i] - hit.position);
				float deltaOmega = An * cosTheta / r2;

				Ray newRay(hit.position + hit.normal * epsilon, controlPoints[i] - hit.position);
				float cosThetaIn = dot(hit.normal, normalize(newRay.dir));

				Hit newHit = firstIntersect(newRay);
				if (newHit.t != -1) {
					if (newHit.material->type == REFLECTIVE)
						outRadiance = outRadiance + trace(newRay, depth + 1) * sun * cosThetaIn * deltaOmega;
				}

				cosTheta = dot(hit.normal, normalize(controlPoints[i] - hit.position));
				if (cosTheta > 0 && !shadowIntersect(newRay)) {	// shadow computation
					outRadiance = outRadiance + sun * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + normalize((controlPoints[0] - hit.position)));
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + sun * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
				outRadiance = outRadiance + sun * cosTheta * hit.material->kd;
			}
		}
		if (hit.material->type == REFLECTIVE)
		{
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad

	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");

}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}
