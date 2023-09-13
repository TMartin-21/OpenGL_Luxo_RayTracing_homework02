//=============================================================================================
// Mintaprogram: Zöld háromszg. Ervenyes 2019. osztol.
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
// Nev    :
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

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) {
		shininess = _shininess;
	}
};

struct Hit {
	float t;
	vec3 position;
	vec3 normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) {
		start = _start;
		dir = normalize(_dir);
	}
};

struct Intersectable {
	Material* material;
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Plane : public Intersectable {
	vec3 position;
	vec3 normal = vec3(0, 0, 1);

	Plane(vec3 _position, Material* _material) {
		position = _position;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 v = ray.start - position;
		hit.t = -v.z / ray.dir.z;
		hit.position = ray.start + ray.dir * hit.t;
		hit.material = material;
		hit.normal = normal;
		return hit;
	}
};

enum Type { SPHERE, PARABOLOID, PLANE, CYLINDER, CYLINDER_PLANE, ARM, TRANSLATE, ROTATE };

mat4 Q_matrix(vec4 v, vec2 h) {
	return mat4(
		v.x, 0, 0, 0,
		0, v.y, 0, 0,
		0, 0, v.z, h.y,
		0, 0, h.x, v.w
	);
}

mat4 Transpose(mat4 m) {
	return mat4(
		m[0][0], m[1][0], m[2][0], m[3][0],
		m[0][1], m[1][1], m[2][1], m[3][1],
		m[0][2], m[1][2], m[2][2], m[3][2],
		m[0][3], m[1][3], m[2][3], m[3][3]
	);
}

mat4 R_yz(float dt) {
	return mat4(
		1, 0, 0, 0,
		0, cosf(dt), -sinf(dt), 0,
		0, sinf(dt), cosf(dt), 0,
		0, 0, 0, 1
	);
}

mat4 R_zx(float dt) {
	return mat4(
		cosf(dt), 0, sinf(dt), 0,
		0, 1, 0, 0,
		-sinf(dt), 0, cosf(dt), 0,
		0, 0, 0, 1
	);
}

mat4 R_xy(float dt) {
	return mat4(
		cosf(dt), -sinf(dt), 0, 0,
		sinf(dt), cosf(dt), 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	);
}

struct TransformMatrix {
	mat4 T;
	Type type;
	TransformMatrix(Type _type, mat4 _T) {
		type = _type;
		T = _T;
	}
};

struct Transform3D {
	TransformMatrix* matrix;
	Transform3D* next;

	void addElement(Transform3D** first, TransformMatrix* _matrix) {
		Transform3D* newElement = new Transform3D();
		newElement->matrix = _matrix;
		newElement->next = NULL;

		Transform3D* last = *first;

		if (*first == NULL)
		{
			*first = newElement;
			return;
		}

		while (last->next != NULL)
		{
			last = last->next;
		}

		last->next = newElement;
	}

	mat4 Inv(mat4 m) {
		return mat4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-m[3][0], -m[3][1], -m[3][2], 1
		);
	}

	mat4 getInverseTransform(Transform3D* list) {
		if (list->next)
		{
			if (list->matrix->type == TRANSLATE)
			{
				return Inv(list->matrix->T) * getInverseTransform(list->next);
			}
			else {
				return list->matrix->T * getInverseTransform(list->next);
			}
		}
		else {
			if (list->matrix->type == TRANSLATE)
			{
				return Inv(list->matrix->T);
			}
			else {
				return list->matrix->T;
			}
		}
	}

	mat4 getTransform(Transform3D* list) {
		if (list->next)
		{
			return list->matrix->T * getTransform(list->next);
		}
		else {
			return list->matrix->T;
		}
	}
};

struct Quadrics : public Intersectable {
	mat4 Q;
	Type type;
	Quadrics* cutPlane;
	Transform3D* list;
	mat4 Q0;

	Quadrics(Type _type, mat4 _Q, Transform3D* _list, Material* _material, Quadrics* _cutPlane = NULL) {
		Q = _Q;
		material = _material;
		type = _type;
		cutPlane = _cutPlane;
		list = _list;
		Q0 = list->getInverseTransform(list) * Q * Transpose(list->getInverseTransform(list));
	}

	vec3 gradf(vec3 r) {
		vec4 g = vec4(r.x, r.y, r.z, 1) * Q0 * 2.0f;
		return vec3(g.x, g.y, g.z);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 S = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec4 D = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);

		float a = dot(D * Q0, D);
		float b = 2.0f * dot(S * Q0, D);
		float c = dot(S * Q0, S);

		float discr = b * b - 4.0f * a * c;

		if (discr < 0)
		{
			return hit;
		}

		float sqrtf_discr = sqrtf(discr);

		float t1 = (-b + sqrtf_discr) / 2.0f / a;
		float t2 = (-b - sqrtf_discr) / 2.0f / a;

		if (type == CYLINDER_PLANE)
		{
			vec3 p1 = ray.start + ray.dir * t1;
			if (p1.x * p1.x + p1.y * p1.y - 5.5f * 5.5f > 0)
			{
				t1 = -1;
			}

			vec3 p2 = ray.start + ray.dir * t2;
			if (p2.x * p2.x + p2.y * p2.y - 5.5f * 5.5f > 0)
			{
				t2 = -1;
			}
		}

		if ((type == ARM || type == PARABOLOID) && cutPlane != NULL)
		{
			vec3 p1 = ray.start + ray.dir * t1;
			vec3 p2 = ray.start + ray.dir * t2;
			vec4 P1 = vec4(p1.x, p1.y, p1.z, 1);
			vec4 P2 = vec4(p2.x, p2.y, p2.z, 1);
			float A = dot(P1 * cutPlane->Q0, P1);
			float B = dot(P2 * cutPlane->Q0, P2);
			if (A > 0)
			{
				t1 = -1;
			}
			if (B > 0)
			{
				t2 = -1;
			}
		}

		if (t1 <= 0 && t2 <= 0)
		{
			return hit;
		}

		if (t1 <= 0)
		{
			hit.t = t2;
		}
		else if (t2 <= 0)
		{
			hit.t = t1;
		}
		else if (t2 < t1)
		{
			hit.t = t2;
		}
		else {
			hit.t = t1;
		}

		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(gradf(hit.position));
		hit.material = material;
		return hit;
	}
};

struct Object3D {
	Transform3D* list = NULL;
	Quadrics* object = NULL;

	void build(vec4 define, Type _type, Quadrics* obj = NULL) {
		Material* material;
		if (_type == SPHERE)
		{
			material = new Material(vec3(0.4f, 0.4f, 0.1f), vec3(2, 2, 2), 70);
		}
		else {
			material = new Material(vec3(0.22f, 0.0f, 0.0f), vec3(2, 2, 2), 70);
		}

		vec2 define2 = vec2(0, 0);
		if (_type == PARABOLOID)
		{
			define2 = vec2(-4, -4);
		}

		object = new Quadrics(_type, Q_matrix(define, define2), list, material, obj);
	}

	void add(TransformMatrix* T) {
		list->addElement(&list, T);
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float windowSize = length(w) * tanf(fov / 2);
		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2 * (X + 0.5f) / windowWidth - 1) + up * (2 * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cosf(dt) + d.y * sinf(dt), -d.x * sinf(dt) + d.y * cosf(dt), 0) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct PointLight {
	vec3 position;
	vec3 Le;
	PointLight(vec3 _Le, vec3 _position) {
		Le = _Le;
		position = _position;
	}
};

const float epsilon = 0.003f;

class Scene {
	std::vector<Object3D> transformObjects;
	std::vector<Intersectable*> objects;
	std::vector<PointLight*> lights;
	Camera camera;
	vec3 La;
	float dt = 0.1f;
public:
	void build() {
		vec3 eye = vec3(0, 70, 30);
		vec3 vup = vec3(0, 0, 1);
		vec3 lookat = vec3(0, 0, 17);
		float fov = 45.0f * M_PI / 180.0f;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.15f, 0.15f, 0.15f);
		lights.push_back(new PointLight(vec3(150, 150, 150), vec3(0, 15, 25)));
		vec3 ks(2, 2, 2);

		Material* plane_material = new Material(vec3(0.3f, 0.2f, 0.1f), ks, 70);
		Plane* plane = new Plane(vec3(0, 0, -0.5f), plane_material);
		objects.push_back(plane);

		Object3D* cylinder = new Object3D;
		Object3D* cutPlane = new Object3D;
		cutPlane->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 0))));
		cutPlane->build(vec4(0, 0, 1, -0.5f), CYLINDER_PLANE);

		cylinder->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 0))));
		cylinder->build(vec4(1, 1, 0, -5.5f * 5.5f), ARM, cutPlane->object);
		objects.push_back(cutPlane->object);
		objects.push_back(cylinder->object);

		Object3D* ballJoint1 = new Object3D;
		ballJoint1->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 0.5f))));
		ballJoint1->add(new TransformMatrix(ROTATE, R_zx(-M_PI / 7.0f))); // -M_PI / 7.0f
		ballJoint1->build(vec4(1, 1, 1, -0.7f * 0.7f), SPHERE);
		objects.push_back(ballJoint1->object);
		transformObjects.push_back(*ballJoint1);

		Object3D* arm1 = new Object3D;
		Object3D* cutplane1 = new Object3D;
		AddTransform(cutplane1);
		cutplane1->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 8))));
		cutplane1->build(vec4(0, 0, 1, -64), CYLINDER_PLANE);
		transformObjects.push_back(*cutplane1);

		AddTransform(arm1);
		arm1->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, -8))));
		arm1->build(vec4(1, 1, 0, -0.4f * 0.4f), ARM, cutplane1->object);
		objects.push_back(arm1->object);
		transformObjects.push_back(*cutPlane);
		transformObjects.push_back(*arm1);

		Object3D* ballJoint2 = new Object3D;
		AddTransform(ballJoint2);
		ballJoint2->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 16))));
		ballJoint2->add(new TransformMatrix(ROTATE, R_zx(M_PI / 1.8f)));
		ballJoint2->build(vec4(1, 1, 1, -0.7f * 0.7f), SPHERE);
		objects.push_back(ballJoint2->object);
		transformObjects.push_back(*ballJoint2);

		Object3D* arm2 = new Object3D;
		Object3D* cutplane2 = new Object3D;
		AddTransform(cutplane2);
		cutplane2->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 8))));
		cutplane2->build(vec4(0, 0, 1, -64), CYLINDER_PLANE);
		transformObjects.push_back(*cutplane2);

		AddTransform(arm2);
		arm2->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, -8))));
		arm2->build(vec4(1, 1, 0, -0.4f * 0.4f), ARM, cutplane2->object);
		objects.push_back(arm2->object);
		transformObjects.push_back(*arm2);

		Object3D* ballJoint3 = new Object3D;
		AddTransform(ballJoint3);
		ballJoint3->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, 16))));
		ballJoint3->add(new TransformMatrix(ROTATE, R_zx(M_PI / 3.0f)));
		ballJoint3->add(new TransformMatrix(ROTATE, R_yz(-M_PI / 4.0f)));
		ballJoint3->build(vec4(1, 1, 1, -0.7f * 0.7f), SPHERE);
		objects.push_back(ballJoint3->object);
		transformObjects.push_back(*ballJoint3);

		Object3D* paraboloid = new Object3D;
		Object3D* cutplane3 = new Object3D;
		AddTransform(cutplane3);
		cutplane3->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, sqrtf(8)))));
		cutplane3->build(vec4(0, 0, 1, -8), CYLINDER_PLANE);
		transformObjects.push_back(*cutplane3);

		AddTransform(paraboloid);
		paraboloid->add(new TransformMatrix(TRANSLATE, TranslateMatrix(vec3(0, 0, -sqrtf(8)))));
		paraboloid->build(vec4(1, 1, 0, 0), PARABOLOID, cutplane3->object);
		objects.push_back(paraboloid->object);

		vec4 p = vec4(0, 0, 0, 1) * ballJoint3->list->getTransform(ballJoint3->list) * TranslateMatrix(vec3(0, 0, 2));
		PointLight* lamp = new PointLight(vec3(150, 150, 150), vec3(p.x, p.y, p.z));
		lights.push_back(lamp);

	}

	void AddTransform(Object3D* object) {
		for (Transform3D* iter = transformObjects[transformObjects.size() - 1].list; iter != NULL; iter = iter->next)
		{
			object->add(new TransformMatrix(iter->matrix->type, iter->matrix->T));
		}
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		for (int Y = 0; Y < windowHeight; Y++)
		{
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++)
			{
				vec3 color = trace(camera.getRay(X, Y));
				image[X + Y * windowWidth] = vec4(color.x, color.y, color.z, 0);
			}
		}
		printf("Rendering time: %d milliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* obj : objects)
		{
			Hit hit = obj->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			{
				bestHit = hit;
			}
		}

		if (dot(ray.dir, bestHit.normal) > 0)
		{
			bestHit.normal = bestHit.normal * (-1);
		}
		return bestHit;
	}

	bool shadowIntersect(Ray ray, float l, Hit hit) {
		for (Intersectable* obj : objects)
		{
			if (obj->intersect(ray).t > 0 && l >= length(obj->intersect(ray).position - hit.position))
			{
				return true;
			}
		}
		return false;
	}

	vec3 trace(Ray ray) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0)
		{
			return La;
		}

		vec3 outRad = hit.material->ka * La;
		for (PointLight* light : lights)
		{
			vec3 lightdir = normalize(light->position - hit.position);
			float l = length(light->position - hit.position);

			Ray shadowRay(hit.position + hit.normal * epsilon, lightdir);
			float cosTheta = dot(hit.normal, lightdir);
			if (cosTheta > 0 && !shadowIntersect(shadowRay, l, hit))
			{
				vec3 LeIn = light->Le / dot(light->position - hit.position, light->position - hit.position);
				outRad = outRad + LeIn * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + lightdir);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0)
				{
					outRad = outRad + LeIn * hit.material->ks * pow(cosDelta, hit.material->shininess);
				}
			}
		}
		return outRad;
	}

	void Animate(float dt) {
		camera.animate(dt);
	}
};

const char* const vertexSource = R"(
	#version 330
	precision highp float;
 
	layout(location = 0) in vec2 cVertexPosition;
	out vec2 texcoord;
 
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1)) / 2;
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);
	}
)";

const char* const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform sampler2D textureUnit;
	in vec2 texcoord;
	out vec4 fragmentColor;
 
	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

Scene scene;
GPUProgram gpuProgram;

class FullScreenTexturedQuad {
	unsigned int vao, vbo, textureID = 0;
public:
	FullScreenTexturedQuad() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };

		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
	}

	void Draw() {
		glBindVertexArray(vao);
		int location = glGetUniformLocation(gpuProgram.getId(), "textureID");
		const unsigned int textureUnit = 0;
		if (location > 0)
		{
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureID);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* f;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	f = new FullScreenTexturedQuad();
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	f->LoadTexture(image);
	f->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
}

void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	scene.Animate(0.1f);
	glutPostRedisplay();
}