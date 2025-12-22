# Copyright Rand Arete @ Ananke 2025
# Licensed under the Apache License, Version 2.0
"""Syntax constraint examples for Kotlin.

This module contains realistic examples of syntax-level constraints that
demonstrate how Ananke's SyntaxDomain masks tokens to enforce Kotlin's
specific syntax patterns, including kotlinx.serialization schema, DSL builders,
and annotation syntax for frameworks like Spring and Ktor.
"""

from __future__ import annotations

try:
    from ..base import ConstraintExample
    from .....spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
        ClassDefinition,
    )
except ImportError:
    from tests.fixtures.constraints.base import ConstraintExample
    from spec.constraint_spec import (
        ConstraintSpec,
        TypeBinding,
        ImportBinding,
        ClassDefinition,
    )

KOTLIN_SYNTAX_EXAMPLES = [
    ConstraintExample(
        id="kt-syn-001",
        name="Kotlinx.serialization Schema",
        description="Enforce valid @Serializable data class structure",
        scenario=(
            "Developer defining a data class with @Serializable annotation for "
            "kotlinx.serialization. The annotation requires specific constraints: "
            "all properties must be serializable types, custom serializers must be "
            "specified with @Serializable(with=...), and @SerialName must be a "
            "compile-time constant string."
        ),
        prompt="""Define a data class for a User with @Serializable annotation for kotlinx.serialization.
Include id, username (with @SerialName for snake_case JSON), and email fields.

""",
        spec=ConstraintSpec(
            language="kotlin",
            structural_tag="serializable_data_class",
            # Regex enforces @Serializable data class pattern
            regex=r"^@Serializable\s+data\s+class\s+\w+\s*\(",
            ebnf=r'''
root ::= user_class | response_class
user_class ::= "@Serializable" nl "data class User(" nl "    val id: String," nl "    @SerialName(\"user_name\")" nl "    val username: String," nl "    val email: String" nl ")"
response_class ::= "@Serializable" nl "data class Response(" nl "    val status: Int," nl "    val data: List<User>" nl ")"
nl ::= "\n"
''',
            imports=[
                ImportBinding(module="kotlinx.serialization", name="Serializable"),
                ImportBinding(module="kotlinx.serialization", name="SerialName"),
            ],
            class_definitions=[
                ClassDefinition(
                    name="User",
                    instance_vars=(
                        TypeBinding(name="id", type_expr="String"),
                        TypeBinding(name="username", type_expr="String"),
                        TypeBinding(name="email", type_expr="String"),
                    ),
                )
            ],
        ),
        expected_effect=(
            "Masks tokens that create invalid serializable structures. Requires "
            "data class (not regular class), val properties (not var unless mutable), "
            "primitive or other @Serializable types for properties. Enforces "
            "@SerialName syntax with string literals."
        ),
        valid_outputs=[
            """@Serializable
data class User(
    val id: String,
    @SerialName("user_name")
    val username: String,
    val email: String
)""",
            """@Serializable
data class Response(
    val status: Int,
    val data: List<User>
)""",
        ],
        invalid_outputs=[
            """@Serializable
class User(val id: String)""",  # Not a data class
            """@Serializable
data class User(var id: String)""",  # Mutable without proper handling
            """@Serializable
data class User(
    @SerialName(getName())
    val id: String
)""",  # SerialName must be constant
        ],
        tags=["syntax", "serialization", "annotations", "data-class", "kotlin"],
        language="kotlin",
        domain="syntax",
    ),
    ConstraintExample(
        id="kt-syn-002",
        name="Kotlin DSL Builder Pattern",
        description="Enforce type-safe builder DSL syntax",
        scenario=(
            "Developer using Kotlin's type-safe builder DSL pattern, commonly seen "
            "in frameworks like kotlinx.html, Ktor routing, Jetpack Compose. "
            "Builder DSLs use lambda with receiver, require @DslMarker to prevent "
            "scope leaks, and specific function syntax for the builder methods."
        ),
        prompt="""Write an HTML DSL builder block using kotlinx.html style.
Use nested lambdas like html { head { title { +"My Page" } } body { ... } }.
Avoid explicit 'this.' references - DSLs use implicit receivers.

""",
        spec=ConstraintSpec(
            language="kotlin",
            structural_tag="dsl_builder",
            expected_type="Html",
            # Regex enforces DSL builder lambda pattern without explicit 'this' or return
            regex=r"^(?:html|routing|compose)\s*\{\s*(?!\s*(?:this\.|return\s))",
            ebnf=r'''
root ::= html_dsl | routing_dsl
html_dsl ::= "html {" nl "    head {" nl "        title { +\"My Page\" }" nl "    }" nl "    body {" nl "        h1 { +\"Welcome\" }" nl "        p { +\"Content here\" }" nl "    }" nl "}"
routing_dsl ::= "routing {" nl "    get(\"/users\") {" nl "        call.respond(users)" nl "    }" nl "    post(\"/users\") {" nl "        val user = call.receive<User>()" nl "        call.respond(user)" nl "    }" nl "}"
nl ::= "\n"
''',
            function_signatures=[],
        ),
        expected_effect=(
            "Masks tokens that don't follow DSL builder patterns. Requires lambda "
            "with receiver syntax ClassName.() -> Unit, enforces proper nesting "
            "without explicit receivers (implicit 'this'), blocks return statements "
            "inside builder lambdas."
        ),
        valid_outputs=[
            """html {
    head {
        title { +"My Page" }
    }
    body {
        h1 { +"Welcome" }
        p { +"Content here" }
    }
}""",
            """routing {
    get("/users") {
        call.respond(users)
    }
    post("/users") {
        val user = call.receive<User>()
        call.respond(user)
    }
}""",
        ],
        invalid_outputs=[
            """html {
    this.head {
        this.title { +"Title" }
    }
}""",  # Explicit 'this' breaks DSL feel
            """html {
    head {
        return title { +"Title" }
    }
}""",  # Return inside builder lambda
        ],
        tags=["syntax", "dsl", "builder", "lambda-with-receiver", "kotlin"],
        language="kotlin",
        domain="syntax",
    ),
    ConstraintExample(
        id="kt-syn-003",
        name="Spring/Ktor Annotation Syntax",
        description="Enforce framework annotation patterns",
        scenario=(
            "Developer writing Spring Boot REST controller or Ktor routing with "
            "proper annotation syntax. Spring requires @RestController, @RequestMapping, "
            "@GetMapping etc. Ktor uses routing DSL. Each has specific syntax "
            "requirements for path parameters, request bodies, and response types."
        ),
        prompt="""Write a Spring Boot REST controller class for a /api/users endpoint.
Use @RestController, @RequestMapping for the base path, and @GetMapping("/{id}") with @PathVariable.

""",
        spec=ConstraintSpec(
            language="kotlin",
            structural_tag="rest_controller",
            # Regex enforces @RestController class with @*Mapping annotations, or Ktor routing DSL
            regex=r"^(?:@RestController[\s\S]*@(?:Get|Post|Put|Delete)Mapping|routing\s*\{[\s\S]*(?:get|post|put|delete)\s*\()",
            ebnf=r'''
root ::= spring_controller | ktor_routing
spring_controller ::= "@RestController" nl "@RequestMapping(\"/api/users\")" nl "class UserController {" nl "    @GetMapping(\"/{id}\")" nl "    fun getUser(@PathVariable id: String): User {" nl "        return userService.findById(id)" nl "    }" nl "}"
ktor_routing ::= "routing {" nl "    route(\"/api/users\") {" nl "        get(\"/{id}\") {" nl "            val id = call.parameters[\"id\"]!!" nl "            call.respond(userService.findById(id))" nl "        }" nl "    }" nl "}"
nl ::= "\n"
''',
            imports=[
                ImportBinding(
                    module="org.springframework.web.bind.annotation",
                    name="RestController",
                ),
                ImportBinding(
                    module="org.springframework.web.bind.annotation",
                    name="GetMapping",
                ),
                ImportBinding(
                    module="org.springframework.web.bind.annotation",
                    name="PathVariable",
                ),
            ],
        ),
        expected_effect=(
            "Masks tokens that don't follow framework annotation patterns. For Spring, "
            "requires proper annotation syntax with path templates in strings, "
            "@PathVariable parameters matching template names. For Ktor, requires "
            "routing DSL with proper call.receive/call.respond patterns."
        ),
        valid_outputs=[
            # Spring style
            """@RestController
@RequestMapping("/api/users")
class UserController {
    @GetMapping("/{id}")
    fun getUser(@PathVariable id: String): User {
        return userService.findById(id)
    }
}""",
            # Ktor style
            """routing {
    route("/api/users") {
        get("/{id}") {
            val id = call.parameters["id"]!!
            call.respond(userService.findById(id))
        }
    }
}""",
        ],
        invalid_outputs=[
            """@RestController
class UserController {
    @GetMapping("/{id}")
    fun getUser(@PathVariable userId: String): User
}""",  # PathVariable name mismatch
            """@GetMapping("/users/{id}")
fun getUser(id: String): User""",  # Missing @PathVariable annotation
        ],
        tags=["syntax", "annotations", "spring", "ktor", "rest", "kotlin"],
        language="kotlin",
        domain="syntax",
    ),
]
